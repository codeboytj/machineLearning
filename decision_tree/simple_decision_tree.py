# 简单的决策树
import math

class DecisionTreeNode:
	"""
	决策树结点，为叶节点的时候表示分类，为普通结点的时候表示样本的某个属性
	"""
    def __init__(self, type, value, children):
        self.type = type
        # 属性名或类名
        self.value = value
        # 子节点
        self.children = children


class SimpleDecisionTree(object):
	"""docstring for SimpleDecisionTree"""
	def __init__(self, arg):
		super(SimpleDecisionTree, self).__init__()
		self.arg = arg

		
	def generate(self, samples, attributes):
		"""
		生成决策树
        :param samples: 训练集，以属性值和类别组成，如(2,3,4,1)，表示属性1,2,3值为2,3,4，而类别为1
        :param attributes: 属性集
        :return: 决策树
		"""
		if inSameCategory(samples):
			# 属于同一类别，标记为根节点返回
			return DecisionTreeNode("叶节点"， samples[0][-1], [])

		if attributes == [] or self.sameAttributes(samples):
			# 属于同一类别，标记为根节点返回,类别为样本中数量最多的类别
			return DecisionTreeNode("叶节点"， mostCategory(samples), [])

		# 选取信息增益最大的属性为划分属性
		root=DecisionTreeNode(""， "", [])
		divAttrIndex=getDivideAttr(samples,attributes,informationEntropy(samples))
		leftAttrs=getLeftAttrs(attributes, divAttrIndex)
		for a,s in devideSamplesByAttrValue(samples,divAttrIndex):
			if s is None or s=[]:
				root.children.append([divAttrIndex,DecisionTreeNode("叶节点"， mostCategory(s), []])
			else:
				root.children.append([divAttrIndex,generate(s,leftAttrs)])
				
	def getLeftAttrs(self, attributes, removeAttr):
		left=[]
		for a in attributes:
			if a!=removeAttr:
				left.append(a)
				
		return left
		
	def devideSamplesByAttrValue(self,samples,attrIndex):
		samplesOfAttr={}
		for sample in samples:
			if samplesOfAttr[sample[attrIndex]] is None:
				samplesOfAttr[sample[attrIndex]]=[]
				samplesOfAttr[0]=sample
			else:
				samplesOfAttr[sample[attrIndex]].append(sample)
				
		return samplesOfAttr

	def getDivideAttr(self,samples,attributes,infoEntropy):
		divAttrIndex=0
		maxInfoGain=None
		for attrIndex in attributes:
			if maxInfoGain is None:
				maxInfoGain=informationGain(samples,infoEntropy,attrIndex)
			else:
				if(maxInfoGain<informationGain(samples,infoEntropy,attrIndex))
					divAttrIndex=attrIndex
					maxInfoGain=informationGain(samples,infoEntropy,attrIndex)

		return divAttrIndex

	def informationGain(self, samples, infoEntropy, attrIndex):
		"""
		计算某一属性对应的信息增益
		"""
		count={}
		samplesOfAttr={}
		for sample in samples:
			if count[sample[attrIndex]] is None:
				count[sample[attrIndex]]=1
				samplesOfAttr[sample[attrIndex]]=[]
				samplesOfAttr[0]=sample
			else:
				count[sample[attrIndex]]=count[sample[attrIndex]]+1
				samplesOfAttr[sample[attrIndex]].append(sample)

		infoGain=infoEntropy
		for attrValue, samplesOfAttrValue in samplesOfAttr.items():
			infoGain=infoGain-count[attrValue]*informationEntropy(samplesOfAttrValue)/samples.length

		return infoGain

	def informationEntropy(self, samples):
		"""
		信息熵
		"""
		count={}
		for sample in samples:
			if count[sample[-1]] is None:
				count[sample[-1]]=1
			else:
				count[sample[-1]]=count[sample[-1]]+1

		probablity=count
		for category,count in count.items():
			probablity=count[category]/samples.length

		infoEntropy=0
		for proValue in probablity.values():
			infoEntropy=infoEntropy+proValue*math.log(proValue,2)

		return -1*infoEntropy


	def inSameCategory(self, samples):
		"""
		返回是否属于同一类别
		"""

		the_category = samples[0][-1]

		for sample in samples:
			if sample[-1] != the_category:
				return False

		return True
	

	def mostCategory(self, samples):
		"""
		返回样本数最多的类别
		"""
		
		times = {}

		for sample in samples:
			if times[sample[-1]] is not None:
				times[sample[-1]]=times[sample[-1]]+1
			else:
				times[sample[-1]]=1

		most=0
		max=0
		for (d,x) in times.items():
			if x>max:
				max=x
				most=d

		return most


	def sameAttributes(self, samples):
		"""
		判断样本的属性值是否相同
		"""
		
		previousAttrValue=samples[0][:-1]

		for sample in samples:
			if previousAttrValue!=sample:
				return False

		return True
