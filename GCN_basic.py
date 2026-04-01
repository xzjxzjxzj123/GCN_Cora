import argparse #argparse = 让你在终端运行程序时传参数，把写死的参数变成可配置的
import os.path as osp #正确处理文件地址，比如在不同操作系统上路径分隔符不同，osp.join()会自动处理这些差异，确保路径正确拼接。osp.dirname(osp.realpath(__file__))获取当前脚本所在的目录，osp.join()将这个目录与其他路径组件（如'..', 'data', 'Planetoid'）拼接成一个完整的路径。
import time #测训练时间 比如start=time.time() end=time.time() print(end-start) 就能知道训练了多久

import torch    #提供了张量（tensor）数据结构和自动求导功能，是构建和训练神经网络的基础工具。
import torch.nn.functional as F #提供了许多常用的损失函数和激活函数等功能，比如交叉熵损失函数（cross_entropy）和ReLU激活函数（relu）等 神经网络常用函数集合

import torch_geometric #GNN库，提供了图神经网络GNN的实现和工具，简化了图数据的处理和模型的构建。torch_geometric.transforms提供了各种图预处理方法，如NormalizeFeatures（特征归一化）和GDC（图扩散卷积）。torch_geometric.datasets提供了常用的图数据集，如Planetoid（Cora、Citeseer、Pubmed）。torch_geometric.logging提供了实验跟踪功能，如init_wandb和log。torch_geometric.nn提供了各种图神经网络层，如GCNConv（图卷积层）。
import torch_geometric.transforms as T #提供了各种图预处理方法，如NormalizeFeatures（特征归一化）和GDC（图扩散卷积）。这些预处理方法可以帮助我们更好地准备图数据，以提高模型的性能和稳定性。
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv

r"""
设置命令行可以输入参数
↓
argparse 解析
↓
args 保存参数
↓
模型和优化器使用这些参数
argparse是python自带标准库，让可以从命令行读脚本
"""
parser = argparse.ArgumentParser()  #创建一个参数解析器对象，准备接收命令行参数.ArgumentParser是argparse模块中的一个类，用于创建一个 参数解析器对象。这个对象可以用来定义程序可以接受的命令行参数，并且在运行时解析这些参数。通过这个对象，我们可以指定参数的名称、类型、默认值以及帮助信息等。当我们运行程序时，ArgumentParser会自动处理命令行输入，并将解析后的参数存储在一个命名空间对象中（通常是args），我们可以通过访问这个对象的属性来获取参数的值。
parser.add_argument('--dataset', type=str, default='Cora')#给程序定义一个可以接收的参数 #双引号、单引号没区别
parser.add_argument('--hidden_channels', type=int, default=16)#16超参数 "--..."是参数名
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')   #action='stroe_true'表示：如果命令行里出现这个参数，就把它设为 True，否则就是 False。#函数调用：add_argument(参数名字,参数行为,帮助说明)#终端输入--help就是显示帮助说明
parser.add_argument('--wandb', action='store_true', help='Track experiment')    #wandb是一个实验跟踪工具，可以记录和可视化训练过程中的各种指标和参数。通过在命令行中添加--wandb参数，我们可以启用wandb的功能，自动记录训练过程中的损失、准确率等指标，以及模型的超参数设置。这对于分析和比较不同实验的结果非常有帮助。
args = parser.parse_args()#读取命令行输入的参数，并存到 args 里。args.dataset 就是命令行输入的 --dataset 参数的值，如果没有输入就用默认值 'Cora'。同样，args.hidden_channels、args.lr、args.epochs、args.use_gdc 和 args.wandb 分别对应其他命令行参数的值。通过属性访问

device = torch_geometric.device('auto')#自动选择运行设备（GPU or CPU） 更常见的是device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#初始化实验记录
#把实验配置登记到wandb里
#本质是调用init_wandb，已经import了
r"""但实际上可能用不到wandb,所以
if args.use_wandb:
    init_wandb(...) 更保险"""
init_wandb(
    name=f'GCN-{args.dataset}',
    lr=args.lr,
    epochs=args.epochs,
    hidden_channels=args.hidden_channels,
    device=device, #记录实验用cuda还是cpu还是gpu #多行参数列表最后一项可以加,
)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')#找到数据在哪里
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())#没有数据就自动下载+加载+预处理
data = dataset[0].to(device)#拿到图数据+放到device里面（为训练做准备）
#dataset = 数据集（可能有多个图）dataset[0] = 第一个图   data = 那张图 这里0就是索引

r"""
__file__:当前这个 .py 文件的路径（字符串）
realpath(__file__):获取当前 .py 文件的绝对路径（字符串）+ 去掉软链接（有的时候是用了快捷方式，这样是为了消除路径假象） 
    绝对路径就是从根目录开始的路径，唯一且不依赖当前位置e.g.C:\Users\xxx\project\train.py # Windows
    相对路径'data/Planetoid' '../data' 相当于“当前文件/当前工作目录
dirname(realpath(__file__)):获取当前 .py 文件所在的目录（路径）（字符串）e.g.project/.py->project
    回到上一级是因为习惯将data和代码放在同一个目录下的不同子文件里面
osp.join(dirname(realpath(__file__)), '..', 'data', 'Planetoid') 拼路径（跨平台安全，比字符串拼接强）
    这里'data'和'Planetoid'是路径组件，就是名字啦
    join只是拼路径，'..' 的含义是后续解释阶段才生效
      osp.join(a, b, c) == b/c 在b是绝对路径的时候丢掉前面部分
    path =从【当前代码文件所在位置】→ 往上走一层→ 进入 data 文件夹→ 进入 Planetoid 文件夹
比如：
realpath得到/home/user/project/code/train.py
dirname得到/home/user/project/code
拼起来得到/home/user/project/code/..
系统自动解析得到（'..'表示跳到上一层目录）
/home/user/project
继续拼得到osp.join(..., 'data', 'Planetoid')
/home/user/project/data/Planetoid
作用：防止写死路径，通过当前代码位置，得到上一级，再得到data



本质写法：其实就是创建一个对象 dataset = Planetoid(root=path, name=args.dataset, transform=...)
    创建对象类似于model = GCN(...) optimizer = Adam(...)
Planetoid是torch_geometric.datasets里的一个类，表示Planetoid数据集。创建这个对象时，我们需要提供一些参数：
    root：数据存储的根目录，这里是path变量，表示数据存储路径，如果数据不存在，Planetoid类会自动下载并存储在这个路径下。
    name：数据集的名称，这里是args.dataset，表示我们要使用哪个数据集（如Cora、Citeseer或Pubmed）。Planetoid类会根据这个名称来加载相应的数据集。
    transform：数据预处理的变换，这里是T.NormalizeFeatures()，表示我们要对数据集中的节点特征进行归一化处理。这个变换会在我们访问数据集中的图数据时(dataset[0]时)自动应用，确保节点特征被归一化到一个合适的范围内，有助于模型的训练和收敛。
path就是数据存储路径，内含意思：没有就自动下载
name告诉程序用哪个数据集
transform是配置，在data=dataset[0]的时候执行

"""
print(data)
print(data.x.shape)
print(data.edge_index.shape)
print(data.edge_index[:, :10])

r"""
data.x表示 2708 个节点（论文)，每个节点 1433 个特征（论文中某个词是否出现）x=[2708,1433]
X =
[ node1 feature vector ]
[ node2 feature vector ]
[ node3 feature vector ]
...

data.edge_index表示 10556 条边，每条边由两个节点组成（论文中某篇文章引用了某篇文章）[2,10556]
edge_index =
[ source nodes ]
[ target nodes ]
不用邻接矩阵，因为2708 × 2708 ≈ 7 million太大了而且大部分是0，所以用edge_index这种稀疏结构表示
比如：
edge_index =
[[0, 1, 2, 3],
[4, 3, 5, 1]]
表示：0 → 4，1 → 3，2 → 5，3 → 1
另外，需要注意的是，这里GCN会默认双向边，核心原因在于目标是节点分类而不是预测引用关系，而引用关系常发生在同一研究领域的文章之间，只要有连接，通常就有语义相似性


data.y表示每个节点的类别标签（论文中每篇文章的研究领域）（我们要预测的）

data.train_mask、data.val_mask、data.test_mask分别表示训练集、验证集和测试集的掩码，指示哪些节点属于哪个集合。它们是布尔张量，长度与节点数量相同（2708），其中True表示该节点属于对应的集合。例如，data.train_mask[i]为True表示第i个节点属于训练集，False表示不属于训练集。通过这些掩码，我们可以在训练和评估模型时选择相应的节点进行计算。
"""

if args.use_gdc:
    transform = T.GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.05),
        sparsification_kwargs=dict(method='topk', k=128, dim=0),
        exact=True,
    )
    data = transform(data)

#定义GCN模型，包含两层图卷积层，每层后面都跟一个ReLU激活函数和一个Dropout层，以防止过拟合。最后一层输出的特征维度等于类别数量，用于节点分类任务。
class GCN(torch.nn.Module):#定义一个GCN类，继承自torch.nn.Module，这是PyTorch中所有神经网络模块的基类，就是继承了Module里面很多好的东西
    def __init__(self, in_channels, hidden_channels, out_channels):#定义GCN类的构造函数，接受输入特征维度、隐藏层维度和输出特征维度作为参数(定义网络结构),初始化GCNConv层是什么意思呢？GCNConv是图卷积层，负责在图结构上进行卷积操作，提取节点的特征信息。GCNConv(in_channels, hidden_channels)表示第一层图卷积层，输入特征维度为in_channels，输出特征维度为hidden_channels；GCNConv(hidden_channels, out_channels)表示第二层图卷积层，输入特征维度为hidden_channels，输出特征维度为out_channels。normalize参数控制是否对邻接矩阵进行归一化处理，这里根据是否使用GDC来决定是否归一化。
        super().__init__()#调用父类的构造函数，确保GCN类正确初始化。super()函数返回父类的一个临时对象，这个对象能够调用父类的方法。在这里，super().__init__()调用了torch.nn.Module的构造函数，完成了GCN类作为一个神经网络模块的初始化工作。这是Python中实现类继承和方法重用的常见方式。
        #创建一个图卷积层GCNConv,Pytorch Geometric已经帮我实现了邻居聚合、线性变换、归一化，可以直接用，我们只需要指定输入输出维度和是否归一化即可。
        #图卷积实际上是个类比名字，卷积神经网络做的是一个像素看周围像素然后提取局部特征，GCN类似的操作是一个节点看周围节点提取特征，所以叫图卷积，GCNConv就是图卷积层的实现。
        #归一化
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(hidden_channels, out_channels,
                             normalize=not args.use_gdc)

    #模型计算流程，得到每个节点的预测类别分布。首先对输入特征进行dropout，随机丢弃一部分特征以防止过拟合。然后通过第一层图卷积层conv1进行特征变换，并应用ReLU激活函数引入非线性。接着再次进行dropout，最后通过第二层图卷积层conv2得到最终的输出特征，这些特征表示每个节点的预测类别分布。输出的形状是[2708, num_classes]，其中2708是节点数量，num_classes是类别数量。
    #这里x是节点特征矩阵（输入）
    #前向传播得到预测分布，之后用于损失函数调整模型参数
    def forward(self, x, edge_index, edge_weight=None):#edge_weight是可选参数，表示边的权重，如果没有提供则默认为None。GCNConv层会根据是否提供edge_weight来决定是否使用边权重进行邻居特征的加权平均。预留edge_weight参数是为了支持GDC等图预处理方法生成的加权边特征。
        x = F.dropout(x, p=0.5, training=self.training)#0.5经验超参数 丢掉一半神经元 #self.training是一个布尔值，表示当前模型是否处于训练模式。当training=True时，dropout会随机丢弃一部分输入特征；当training=False时，dropout不会丢弃任何特征，而是直接返回输入。这是因为在评估或测试阶段，我们希望使用所有的特征来获得稳定的预测结果，而在训练阶段，我们希望通过丢弃特征来防止过拟合。
        x = self.conv1(x, edge_index, edge_weight).relu() #线性变换发生在图卷积层conv1中，具体来说是在GCNConv的实现中。当我们调用self.conv1(x, edge_index, edge_weight)时，GCNConv会执行以下操作：1.邻居聚合：GCNConv会根据edge_index来聚合每个节点的邻居特征，计算出每个节点的新特征表示。2.线性变换：GCNConv会将聚合后的特征乘以权重矩阵（这是GCNConv层的参数），进行线性变换。3.归一化（如果normalize=True）：GCNConv会对邻接矩阵进行归一化处理，以稳定训练过程。4.返回新的节点特征表示。之后，我们调用.relu()来对卷积层的输出应用ReLU激活函数，引入非线性，使模型能够学习更复杂的特征表示。
        x = F.dropout(x, p=0.5, training=self.training)#每一层都有可能过拟合，所以两次dropout #特征变换发生在
        x = self.conv2(x, edge_index, edge_weight)#输出层没有非线性 #经典的两层GCN设计：Input → GCNConv + ReLU → Dropout → GCNConv → Output
        return x 

#定义一个GCN的实例
model = GCN(
    in_channels=dataset.num_features,#numfeatures表示输入维度，PyG 自动从数据中提取这些信息 #print(dataset.num_features)  # 自动计算：从 data.x.shape[1] 得到#print(dataset.num_classes)   # 自动计算：从 data.y.unique().size(0) 得到
    hidden_channels=args.hidden_channels,
    out_channels=dataset.num_classes,#输出维度
).to(device)

#创建一个optimizer对象，使用Adam优化算法来更新模型参数。这里我们分别为两层图卷积层设置了不同的权重衰减（weight_decay）值，第一层设置为5e-4，第二层设置为0。这是因为在GCN中，第一层通常需要更多的正则化来防止过拟合，而第二层则不需要太多的正则化。学习率（lr）也通过命令行参数进行设置。
#定义了2个参数组，分别对应模型的两层图卷积层conv1和conv2
#parameters()方法返回一个生成器，迭代器会返回模型中所有可训练参数的张量。通过指定不同的参数组，我们可以为不同的层设置不同的优化超参数（如权重衰减）。在这个例子中，我们为conv1层设置了权重衰减5e-4，而conv2层没有权重衰减（weight_decay=0）。这种做法有助于防止第一层过拟合，同时允许第二层更灵活地学习特征表示。
#conv1:loss+5e-4*||w||^2  conv2:loss+0*||w||^2
#W = W - lr * gradient lr是学习率(每次更新走多大一步），gradient是损失函数关于参数的梯度.
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),#等价于"params": model.conv1.parameters(),"weight_decay": 5e-4
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=args.lr)  # Only perform weight-decay on first convolution.

#训练函数，执行一次训练迭代，包括前向传播forward、计算损失loss、反向传播backward和参数更新。首先将模型设置为训练模式（model.train()），然后清零优化器的梯度（optimizer.zero_grad()）。接着将输入数据传入模型，得到输出out。计算交叉熵损失函数，只使用训练集节点进行计算。最后进行反向传播（loss.backward()）和参数更新（optimizer.step()）。函数返回当前的损失值。
def train():
    model.train()#将模型设置为训练模式，这会启用dropout等训练特定的行为。
    optimizer.zero_grad()#清零优化器的梯度，防止梯度累积。PyTorch中的默认行为是梯度会累积在参数的.grad属性中，所以在每次训练迭代开始时需要清零梯度，以确保每次迭代的梯度计算都是独立的。
    out = model(data.x, data.edge_index, data.edge_attr)#把数据输入模型，得到输出out，out是一个张量，表示每个节点的预测类别分布。out的形状是[2708, num_classes]，其中2708是节点数量，num_classes是类别数量。每行表示对应节点的预测类别分布，即每个元素表示该节点属于某个类别的概率（是打分，不是最终预测结果）。numclasses是dataset.num_classes，表示数据集中类别的数量，比如Cora数据集有7个类别，所以num_classes=7。
    #上面out是如何调用了forward:用了对象的call方法，实际上是model->__call__->(一些内部操作）->forward,
    #edge_attr是边特征，没有就自动none
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])#分类任务最常用的损失函数 且 只用train_mask对应的节点进行计算损失，因为我们只用训练集来更新模型参数。out[data.train_mask]表示选择训练集节点的预测类别分布，data.y[data.train_mask]表示选择训练集节点的真实类别标签。交叉熵损失函数会比较预测类别分布和真实类别标签之间的差异，计算出一个标量损失值，表示模型在训练集上的表现。
    loss.backward()#反向传播算法，用于计算损失函数关于模型参数的梯度。这个步骤会自动计算出每个参数的梯度值，存储在参数的.grad属性中。
    optimizer.step()#更新模型参数。这个步骤会根据计算出的梯度值，按照优化算法（这里是Adam（会计算梯度平均和梯度平方平均）的规则来调整模型参数的值，以最小化损失函数。
    return float(loss.detach())#返回当前的损失值。loss.detach()会返回一个新的张量，与原来的loss共享数据但不需要梯度计算，这样可以避免在后续的计算中对loss进行反向传播。float()将张量转换为Python的浮点数，方便打印和记录损失值。
    #核心原因：loss 包含整个反向传播的计算图，直接返回会保留所有中间结果，浪费内存.detach只保留最终张量（数值）

@torch.no_grad()#关闭梯度计算，节省内存和计算资源，因为在测试阶段我们不需要计算梯度。
def test():
    model.eval()#将模型设置为评估模式，这会禁用dropout等训练特定的行为.
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)#把数据输入模型，得到输出out，然后使用argmax(dim=-1)来获取每个节点的预测类别标签。argmax(dim=-1)会返回沿着最后一个维度（类别维度）最大值的索引，即每个节点预测的类别标签。pred的形状是[2708]，其中每个元素表示对应节点的预测类别标签。
    #这里model输出一个矩阵2708*7，每一行是一个节点属于7个类别的“分数”
    #argmax(dim=-1)沿着最后一个维度找最大值的位置(类别维度)，返回一个长度为2708的一维张量，每个元素是对应节点预测的类别标签（0到6之间的整数，表示7个类别中的一个）。
    """这里重新forward了：1.参数更新了，重新跑一遍2.相当于model已经OK了，不要dropout在全部数据上测一下"""
    accs = []#accs是一个空列表，用于存储训练集、验证集和测试集的准确率。我们将通过循环计算每个集合的准确率，并将结果添加到这个列表中。最后，函数会返回这个列表，包含三个元素，分别对应训练集、验证集和测试集的准确率。
    for mask in [data.train_mask, data.val_mask, data.test_mask]:#mask是一个布尔张量，长度与节点数量相同（2708），其中True表示该节点属于对应的集合（训练集、验证集或测试集）。我们通过这个mask来选择对应集合的节点进行准确率计算。如果mask为[1,0,1],x[mask]就会选择x中的第1和第3个元素进行计算。
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))#Pytorch里所有运算默认返回tensor，所以需要int()把结果转换成Python的整数。pred[mask]表示选择对应集合的节点的预测类别标签，data.y[mask]表示选择对应集合的节点的真实类别标签。通过(pred[mask] == data.y[mask])可以得到一个布尔张量，表示每个节点的预测是否正确。sum()函数会计算出预测正确的节点数量，而mask.sum()会计算出该集合中的节点总数。最终，准确率就是预测正确的节点数量除以总节点数量。
    return accs# # 每次循环添加一个数值到列表.循环结束后，accs 有三个元素 ; accs = [训练集准确率, 验证集准确率, 测试集准确率]


best_val_acc = test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    times.append(time.time() - start)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
#一共epochs次，每次train更新模型，保留val代表最高泛化能力（至少更大概率是最高泛化能力），提示过拟合，取这一次对应的epoch的test作为最终结果
#争议：模型看了test,但可接受，更严谨，最终取了val最大的epoch对应的参数，再用在test身上（就是加一行保存参数的代码）

r"""
tensor 用于表示向量空间中的元素（nD数组），然后承载向量、矩阵的运算，这就是神经网络在做的事情
比如：y = torch.matmul(W, x) + b
x是输入向量，W是权重矩阵，b是偏置向量，y是输出向量


tensor优点：支持自动求导 和 矩阵运算 和 GPU加速 
关于自动求导：必须知道参数往哪个方向改loss会下降，所以需要知道dloss/dw(w是参数)    
所以深度学习框架本质是一个自动求导框架


神经网络工具：torch.nn（一个神经网络库）帮助我们
1.定义网络结构：输入输出向量的维度，和层之间如何连接（即为线性变换+激活函数（激活函数是非线性函数））
2.管理参数：权重矩阵和偏置向量，自动更新参数
3.提供常见的层和损失函数：如线性层、卷积层、循环神经网络层、Transformer层等，以及交叉熵损失函数等

几个常见的神经网络层：
1.nn.linear(3,2) 数学形式：y=W*X+b  W是权重矩阵，b是偏置向量
2.nn.conv2d 卷积层，常用于图像处理
3.nn.LSTM 循环神经网络层，常用于处理序列数据，记住过去的信息
4.nn.Transformer 自然语言处理 Attention机制和并行计算


为什么深度学习都能写成y=f(WX+b)的形式：任何复杂函数都可以用很多“线性变换 + 非线性函数”叠加来近似。
其中WX:重新组合特征，每个新特征都是旧特征的重新线性组合
权重矩阵通过神经元实现：
Linear(20,10)20个输入特征，10个输出特征，每个输出特征都是20个输入特征的线性组合（也是10个神经元），每个神经元有20个权重，再加上偏置（10个输出特征对应10个偏置），所以总共有20*10+10=210个参数



GCNConv 在内部到底怎么用 edge_index 聚合邻居特征:
每个节点的新特征 = 自己 + 邻居节点特征 的加权平均

GCNConv的核心数学公式：
h_i' = σ( W * (h_i + Σ_{j∈N(i)} h_j / sqrt(d_i * d_j)) )
其中h_i是节点i的输入特征，h_i'是节点i的输出
H^(l+1) = σ(Â H^(l) W^(l))  
    H^(l) = 第 l 层节点表示 W^(l) = 第 l 层参数 σ = 激活函数 Â = 归一化邻接矩阵
    邻接矩阵A*节点特征矩阵*权重矩阵
        A_ij = 1   如果 i 和 j 有边 A_ij = 0   如果没有边->邻接矩阵乘特征矩阵= 自动聚合邻居特征 W特征变换 ReLU激活函数引入非线性

1.遍历所有边找到邻居
2.对每个节点i，计算邻居特征的加权平均，权重是1/sqrt(d_i * d_j)，其中d_i和d_j是节点i和邻居j的度数
3.将节点i的输入特征h_i与邻居特征的加权平均相加
4.对结果进行线性变换W（乘以权重矩阵，1433维变成16维），和非线性激活σ，得到节点i的新特征h_i'
"""
"""
小训练：
1.
import torch
x = torch.tensor([
    [1, 2],
    [3, 4],
    [5, 6]
])
mask = torch.tensor([True, False, True])
print(x[mask])
输出： tensor([[1, 2],
             [5, 6]])
按照mask选行

2.
out = torch.tensor([
    [0.1, 0.9],
    [0.8, 0.2],
    [0.3, 0.7]
])
out.argmax(dim=1) =?
ANS: tensor([1, 0, 1])
argmax(dim=1)沿着第1维（列）找最大值的位置，返回一个长度为3的一维张量，每个元素是对应行最大值的索引（0或1），表示每个节点预测的类别标签。
（第0维是行，第1维是列）

3.
train_mask = [True, False, True]
y = [1, 0, 1]
ANS: y[train_mask] = [1, 1]
只选择train_mask为True的元素进行计算

4.1
x = [
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
]
x[:,0]= [1, 3, 5, 7] #:所有行 0第0列
x[1:3]=
[
  [3, 4],
  [5, 6]
]
python左闭右开
4.2
mask = [True, False, True, False]
x[mask,0]= [1, 5] 先按照mask选行，再选第0列
4.3
out = [
    [2.0, 1.0, 0.5],
    [0.1, 0.2, 0.9],
    [3.0, 1.0, 2.0]
]
out[:,1]= [1.0, 0.2, 1.0] 选第1列
out.argmax(dim=1)= [0, 2, 0] 选每行最大值的索引
out.argmax(dim=1)[mask]= [0, 0] 先选每行最大值的索引，再按照mask选行
x[:, 0:2] 二维矩阵 → shape = (n, 2)

"""