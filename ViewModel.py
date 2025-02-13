from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from sklearn.cluster import KMeans
from layers import ZINBLoss, MeanAct, DispAct
from torch.utils.data import DataLoader, TensorDataset
import warnings
from evaluation import evaluate
from torch.nn.parameter import Parameter
import math
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)



class Clustering_Layer(nn.Module):
    def __init__(self, inputDim, n_cluster):
        super(Clustering_Layer, self).__init__()
        hidden_layers = [nn.Linear(inputDim, 32), nn.ReLU()]
        hidden_layers.append(nn.BatchNorm1d(num_features=32))
        self.hidden = nn.Sequential(*hidden_layers)
        self.withoutSoft = nn.Linear(32, n_cluster)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        hidden = self.hidden(x)
        withoutSoftMax = self.withoutSoft(hidden)
        output = self.output(withoutSoftMax)
        return output, withoutSoftMax, hidden


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim, num_views):
        super(AdditiveAttention, self).__init__()
        self.input_dim = input_dim
        self.num_views = num_views

        # 每个视图的权重参数
        self.w = nn.Parameter(torch.randn(num_views, input_dim))  # [num_views, input_dim]
        self.b = nn.Parameter(torch.randn(num_views, 1))  # [num_views, 1]
        self.cuda()

    def forward(self, zi_stack):
        """
        :param zi_stack: 输入张量，形状为 [num_views, n, input_dim]
        :return: 加权融合结果，形状为 [n, input_dim]
        """
        # 将输入列表转换为张量，形状：[num_views, n, input_dim]
        zi_stack = torch.stack(zi_stack, dim=0)  # 堆叠视图

        num_views, n, input_dim = zi_stack.shape

        # 计算 sm（注意力得分），逐视图计算
        sm = []
        for view in range(self.num_views):
            # 对每个视图的输入计算权重
            sm_view = torch.tanh(
                torch.matmul(zi_stack[view], self.w[view].unsqueeze(1)) + self.b[view]
            )  # [n, 1]
            sm.append(sm_view)

        # 合并 sm 得分，形状：[num_views, n, 1]
        sm = torch.stack(sm, dim=0)

        # 对每个视图进行权重归一化，形状：[num_views, n, 1]
        sm_exp = torch.exp(sm)
        ai = sm_exp / sm_exp.sum(dim=0, keepdim=True)  # 对视图维度归一化

        # 加权求和，形状：[n, input_dim]
        z_fusion = torch.sum(ai * zi_stack, dim=0)  # 加权求和，沿视图维度合并

        return z_fusion  # 返回形状：[n, input_dim]


def buildNetwork(layers, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


def make_qp(x, centroids):
    q = 1.0 / (1.0 * 0.001 + torch.sum(torch.pow(x.unsqueeze(1).cuda() - torch.tensor(centroids).cuda(), 2), 2))
    q = (q.t() / torch.sum(q, 1)).t()
    weight = q ** 2 / q.sum(0)
    p = (weight.t() / weight.sum(1)).t().data
    return q, p


class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, n_z):
        super(ClusteringLayer, self).__init__()
        self.centroids = Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)

    def forward(self, x):
        q = 1.0 / (1 + torch.sum(
            torch.pow(x.unsqueeze(1) - self.centroids, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        return q


class SingleViewModel(nn.Module):
    def __init__(self, batch_size, size_factor, input_dim, z_dim, n_clusters, lr, encodeLayer=[], decodeLayer=[],
                 views=[],
                 activation="relu", sigma=1., alpha=1., gamma=1., device="cuda"):
        super(SingleViewModel, self).__init__()
        torch.set_default_dtype(torch.float64)

        self.size_factor = size_factor
        self.batch_size = batch_size,
        self.lr = lr,
        self.z_dim = z_dim  # 32
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation)  ###2000-256-64
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation)  ###32-64-256
        self.decoder1 = buildNetwork([z_dim] + decodeLayer, activation=activation)  ###32-64-256
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())
        self.mu = nn.Parameter(torch.Tensor(n_clusters, z_dim))  ###生成一个10*32的质心坐标，即每个簇的坐标都是32维
        self.clusteringLayer = ClusteringLayer(n_clusters, z_dim)
        self.zinb_loss = ZINBLoss().to(self.device)
        self.to(device)

    def pretrain_autoencoder(self, viewIndex, batch_size, x, size_factor, epoch, lr):

        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(size_factor))

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 每批batch_size=256个，比如268个细胞就只有两批
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        with tqdm(range(epoch), desc='Pre view %d' % viewIndex) as tbar:
            for epoch in tbar:

                for batch_idx, (x_batch, sf_batch) in enumerate(dataloader):
                    x_tensor = x_batch.cuda()
                    size_factor = sf_batch.cuda()
                    z0, q, mean_tensor, disp_tensor, pi_tensor, x0, _ = self.forward(
                        x_tensor)  ###进入上面的45行的forward,输入就一个x_tensor，输出三个

                    loss = self.zinb_loss(x=x_tensor, mean=mean_tensor.cuda(), disp=disp_tensor.cuda(),
                                          pi=pi_tensor.cuda(),
                                          scale_factor=size_factor)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tbar.set_postfix(loss=loss.item())

        return z0, q

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def encodeBatch(self, X, batch_size):
        X = torch.Tensor(X)
        encoded = []  ###encoded是一个空列表
        qlist = []
        num = X.shape[0]  ####num=268即细胞数目
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))  ###每批次256，算算要多少批，这里是268/256=2
        for batch_idx in range(num_batch):  ###对每一批
            xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]  ###取该批次的数据
            inputs = Variable(xbatch).to(self.device)
            z, q, _, _, _, _, _ = self.forward(inputs)  ###对数据进行编码2000-256-64-32得到32维的z
            encoded.append(z.data)  ###将这256个32维的数据存在encoded列表中，作为它的一个元素，回到for继续，直到所有批次存完，最后encoded列表中存放的是所有细胞的32维的编码
            qlist.append(q.data)

        qlist = torch.cat(qlist, dim=0)  ###将encoded列表的所有元素拼接起来，成为一个(num,32)的tensor
        encoded = torch.cat(encoded, dim=0)  ###将encoded列表的所有元素拼接起来，成为一个(num,32)的tensor
        return encoded.to(self.device), qlist.to(self.device)

    def forward(self, x):  ###作用是对数据进行预训练，得到相应的中间变量，z0, q是用原始的x得到的，_mean, _disp, _pi是用加了噪声的x得到的
        x = x.cuda()
        h = self.encoder(x + torch.randn_like(x) * self.sigma)  ##将输入加噪声以后进行编码，2000-256-64
        z = self._enc_mu(h)  ####64-32
        h = self.decoder(z)  ####32-64-256

        h1 = self.decoder1(z)  ####32-64-256
        h = (h1 + h) / 2  ###求平均是为了减小波动

        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)  ##将输入直接进行编码，2000-256-64
        z0 = self._enc_mu(h0)  ####64-32
        x0 = self.decoder(z0)
        q = self.soft_assign(z0)  ###直接对隐藏层z0求软分配，度量的是z0和self.mu的相似性
        q1 = 0
        return z0, q, _mean, _disp, _pi, x0, q1


class MultiViewModel(nn.Module):

    def __init__(self, z_dim, n_clusters, labels, viewNumber, size_factor, batch_size, encodeLayer, decodeLayer, views,
                 lr,
                 activation="relu", sigma=2.5, alpha=1., gamma=1., device="cuda"):
        super(MultiViewModel, self).__init__()

        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.labels = labels
        self.viewNumber = viewNumber
        self.size_factor = size_factor
        self.batch_size = batch_size
        self.encodeLayer = encodeLayer
        self.decodeLayer = decodeLayer
        self.views = views
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.lr = lr
        self.cluster_projector = nn.Sequential(  #####聚类投影，z_dim维投影到n_clusters维
            nn.Linear(z_dim, n_clusters),  ####32-10
            nn.Softmax(dim=1))
        self.to(device)

        aes = []
        for viewIndex in range(viewNumber):
            aes.append(SingleViewModel(self.batch_size, size_factor[viewIndex],
                                       views[viewIndex].shape[1], z_dim, n_clusters, self.lr, encodeLayer,
                                       decodeLayer, views,
                                       activation, sigma, alpha, gamma, device))

        self.aes = nn.ModuleList(aes)
        self.cluster_layer = Clustering_Layer(z_dim, n_clusters)
        self.zinb_loss = ZINBLoss().to(self.device)
        self.cuda()


    def pretrain(self, x, epoch, lr):

        encodelist = []
        for viewIndex in range(self.viewNumber):
            self.aes[viewIndex].pretrain_autoencoder(viewIndex, self.batch_size, x[viewIndex],
                                                     self.size_factor[viewIndex], epoch, lr)
            encoded, _ = self.aes[viewIndex].encodeBatch(x[viewIndex], self.batch_size)
            encodelist.append(encoded)

        input_size = encodelist[0].shape[1]

        # 初始化模型
        attention_model = AdditiveAttention(input_size, self.viewNumber)

        # 前向计算
        z_fusion = attention_model(encodelist)  # 输出形状：[n, input_dim]

        return z_fusion

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(
                torch.sum(target * torch.log(target / pred), dim=-1))  ####这里求了均值，所以后面会* len(inputs)

        p = np.asarray(p.cpu().detach().numpy(), dtype=np.float64)
        q = np.asarray(q.cpu().detach().numpy(), dtype=np.float64)
        p = p / np.sum(p)
        q = q / np.sum(q)
        kldloss = kld(torch.Tensor(p), torch.Tensor(q))
        return kldloss * 1.0  ####self.gamma=1.0


    def pre(self, enlist, y):
        z_fusion = enlist

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100)

        self.y_pred = kmeans.fit_predict(z_fusion)  # kmeans聚类预测标签

        # 对第一次特征融合进行Kmeans预测
        # if y is not None:
        acc, nmi, ari, homo, comp = evaluate(y, self.y_pred)
        print('Initializing k-means: ACC= %.4f, ARI= %.4f, NMI= %.4f' % (acc, ari, nmi))

        return acc, ari, nmi

    def fit(self, z_fusion, X, input_size, views, size_factor, n_clusters, y, lr, epoch, tol, batch_size, save_path, b):

        # 初始化
        viewNum = len(views)
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95,
                                   weight_decay=0.001)
        kmeans = KMeans(n_clusters=n_clusters, n_init=100)

        print(z_fusion.shape)
        print(z_fusion)

        self.y_pred = kmeans.fit_predict(z_fusion.cpu().detach().data.numpy())  # kmeans聚类预测标签
        self.y_pred_last = self.y_pred  ###备份一下kmeans聚类得到预测标签
        self.last_mu = kmeans.cluster_centers_

        # 对第一次特征融合进行Kmeans预测
        if y is not None:
            acc, nmi, ari, homo, comp = evaluate(y, self.y_pred)
            print('Initializing k-means: ACC= %.4f, ARI= %.4f, NMI= %.4f' % (acc, ari, nmi))

        self.train()
        num = z_fusion.shape[0]  ###细胞数
        num_batch = int(math.ceil(1.0 * z_fusion.shape[0] / batch_size))  ###算下需要多少批次
        lst = []  #####创建一个空列表，用于存放指标
        pred = []  #####创建一个空列表，用于存放预测标签
        best_ari = 0.0  # 存放最好的ari指标

        # 训练多个epoch
        with tqdm(range(epoch), desc="fit") as tbar:
            for epoch in tbar:
                z_fusion = None

                output = self.forward(views)

                z_list = []
                for viewIndex in range(self.viewNumber):
                    z_list.append(output[viewIndex][0])
                # 融合特征
                input_size1 = output[0][0].shape[1]
                num1 = output[0][0].shape[0]
                encodelist1 = []
                for viewIndex in range(self.viewNumber):
                    encodelist1.append(output[viewIndex][0])

                # 初始化模型
                model_a_w = AdditiveAttention(input_size1, self.viewNumber)

                # 前向计算
                z_fusion = model_a_w(encodelist1)  # 输出形状：[n, input_dim]

                kmeans = KMeans(self.n_clusters, n_init=100)
                kmeans.fit_predict(z_fusion.cpu().detach().numpy())
                q, p = make_qp(z_fusion, kmeans.cluster_centers_)

                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()  ###用软标签q得到预测标签

                acc, nmi, ari, homo, comp = evaluate(y, self.y_pred)  ##算软标签q的指标

                pred.append(self.y_pred)
                zhibiao = (acc, nmi, ari, homo, comp)
                lst.append(zhibiao)

                if best_ari < ari:  ###如果当前得到的ari比最优的ari大，说明当前的更好，就把当前的存起来，最终保存的是训练次数中最优的ari
                    best_ari = ari
                    torch.save({'latent': z_fusion, 'q': q, 'p': p}, save_path)

                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / output[viewIndex][0].shape[0]
                self.y_pred_last = self.y_pred

                if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break

                # train 1 epoch for clustering loss

                for batch_idx in range(num_batch):

                    recon_loss = 0.
                    kl_loss = 0.
                    qlist = []
                    views_batch = []
                    for view in views:
                        views_batch.append(view[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)])
                    output = self.forward(views_batch)
                    q_zmu, _, _ = self.cluster_layer(
                        z_fusion[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)])
                    p_patch = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]

                    for viewIndex in range(len(views)):
                        rawinputs = torch.Tensor(views_batch[viewIndex]).cuda()
                        sfinputs = torch.Tensor(size_factor[viewIndex][
                                                batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]).cuda()
                        mean = output[viewIndex][2].cuda()
                        disp = output[viewIndex][3].cuda()
                        pi = output[viewIndex][4].cuda()

                        recon_loss = recon_loss + self.zinb_loss(rawinputs, mean, disp, pi, sfinputs)
                        q_temp = output[viewIndex][1]
                        qlist.append(q_temp)

                    for viewIndex in range(self.viewNumber):
                        kl_loss = kl_loss + self.cluster_loss(p_patch, qlist[viewIndex])
                    recon_loss = recon_loss / len(views)

                    loss = recon_loss + b * kl_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                tbar.set_postfix(recon_loss=recon_loss.item(),
                                 acc=acc,
                                 ari=ari, nmi=nmi)
                tbar.update(1)

        print('Start pred.')

        cunari = []  #####初始化
        for j in range(len(lst)):  ###j从0到num_epochs-1
            aris = lst[j][2]
            cunari.append(aris)
        max_ari = max(cunari)
        maxid = cunari.index(max_ari)
        optimal_pred = pred[maxid]

        final_acc, final_nmi, final_ari, final_homo, final_comp = evaluate(y, optimal_pred)
        return final_acc, final_nmi, final_ari, final_homo, final_comp

    def forward(self, x):
        outputs = []
        for viewIndex in range(len(x)):
            x_temp = torch.Tensor(x[viewIndex]).cuda()
            outputs.append(self.aes[viewIndex](x_temp))
        return outputs

    def encodeBatch(self, views):  ####encodeBatch的作用是对输入进行编码得到编码的嵌入向量，输入就是原始的x，
        # 每次只处理一个批次的数据，将数据按批次处理，避免超显存
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        output = self.forward(views)
        z_fusion = None
        # 融合特征
        for viewIndex in range(len(views)):
            z_temp = output[viewIndex][0]
            if (viewIndex) == 0:
                z_fusion = z_temp
            else:
                z_fusion = torch.cat((z_fusion, z_temp), dim=1)

        return z_fusion
