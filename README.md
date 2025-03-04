# 分布式复合优化

这个软件包提供了分布式复合优化的算法。

## 安装
### 1. 安装 Git（可选但推荐）
建议执行这一步，无论下一步采取什么方法下载源代码，因为管理通信的包是放在远端 GitHub 里，通过 Git 直接安装会更方便。

#### Windows:

请按照[知乎](https://zhuanlan.zhihu.com/p/242540359)上的说明进行操作。

#### Linux:

```bash
sudo apt-get update
sudo apt-get install git
```

#### macOS:

```bash
brew install git
```

### 2. 下载源代码
#### 方法一
打开终端并运行以下命令：
```bash
git clone https://github.com/rui-huang-opt/dco.git
```

**注意：**
由于目前仍是私人仓库，下载源码需联系 R.Huang@lboro.ac.uk 获取 token。

#### 方法二
通过压缩包解压。

**注意：**
如果之前未安装 Git，请额外前往 https://github.com/rui-huang-opt/gossip ，进入网页后点击右上角绿色按钮 Code，在展开的菜单里选择 Download ZIP，下载并解压得到源代码包 gossip。

### 3. 创建并激活虚拟环境（可选但推荐）
创建虚拟环境可以帮助你将项目代码与全局的 Python 环境隔离开，防止包之间的冲突，因此推荐这样做。
推荐在保存了源代码的文件夹
```plaintext
你保存了源代码的目录/
├─ .venv
├─ dco
```
或源代码根目录下创建
```plaintext
你保存了源代码的目录/
├─ dco/
│  ├─ .venv
|  ├── ...
```
在进入相应目录后，执行：

#### Windows:

```bash
python -m venv env
.\env\Scripts\activate
```

#### Linux and macOS:

```bash
python3 -m venv env
source env/bin/activate
```

### 4. 安装包及依赖
在激活的虚拟环境中，进入 dco 包的根目录，并执行：

```bash
pip install .
```
**注意：**
如果之前你没有安装 Git，请按照之前的教程下载 gossip 包的源码，并同样在同一个环境（虚拟环境或全局环境，只要与dco在同一个之下即可）下进入 gossip 根目录运行
```bash
pip install .
```

### 5. 运行测试
确保一切正常工作，运行以下命令进行测试：

```bash
cd tests
python ridge_regression.py
```
结果会保存在同目录下的 figures 目录里。

### 6. 使用
对于问题