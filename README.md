# DWFRGCN

1、本项目在lightGCN的基础上更改，环境配置可依据一下LightGCN部分；
2、DWFRGCN所需Lib包太大暂时无法上传，下载后可在LightGCN的tensorflow版本代码的venv文件夹下自行获取；
3、如需在本项目中调试LightGCN模型，在项目中的在/utility/parser.py中切换默认模型为lightgcn即可。
4、运行命令：
Gowalla dataset
python DWFRGCN.py --dataset gowalla --regs [1e-4] --embed_size 64 --layer_size [64,64,64,64,64,64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000

Yelp2018 dataset
python DWFRGCN.py --dataset yelp2018 --regs [1e-4] --embed_size 64 --layer_size [64,64,64,64,64,64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000

Amazon-book dataset
python DWFRGCN.py --dataset amazon-book --regs [1e-4] --embed_size 64 --layer_size [64,64,64,64,64,64,64,64] --lr 0.001 --batch_size 8192 --epoch 1000

NOTE : the duration of training and testing depends on the running environment.

5：以下为基线模型LightGCN相关信息：       
LightGCN
This is our Tensorflow implementation for our SIGIR 2020 paper:

Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, Paper in arXiv.

Contributors: Dr. Xiangnan He (staff.ustc.edu.cn/~hexn/), Kuan Deng, Yingxin Wu.

(We also provide Pytorch implementation for LightGCN : https://github.com/gusye1234/LightGCN-PyTorch. Contributors: Jianbai Ye.)

Introduction
In this work, we aim to simplify the design of GCN to make it more concise and appropriate for recommendation. We propose a new model named LightGCN, including only the most essential component in GCN—neighborhood aggregation—for collaborative filtering.

Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:

tensorflow == 1.11.0
numpy == 1.14.3
scipy == 1.1.0
sklearn == 0.19.1
cython == 0.29.15
C++ evaluator
We have implemented C++ code to output metrics during and after training, which is much more efficient than python evaluator. It needs to be compiled first using the following command.

python setup.py build_ext --inplace
After compilation, the C++ code will run by default instead of Python code.

Dataset
We provide three processed datasets: Gowalla, Yelp2018 and Amazon-book.

train.txt

Train file.
Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.
test.txt

Test file (positive instances).
Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.
Note that here we treat all unobserved interactions as the negative instances when reporting performance.
user_list.txt

User file.
Each line is a triplet (org_id, remap_id) for one user, where org_id and remap_id represent the ID of the user in the original and our datasets, respectively.
item_list.txt

Item file.
Each line is a triplet (org_id, remap_id) for one item, where org_id and remap_id represent the ID of the item in the original and our datasets, respectively.
Efficiency Improvements:
Parallelized sampling on CPU
C++ evaluation for top-k recommendation

=======
