from math import log
import operator
import treePlotter
from collections import Counter


pre_pruning = False
post_pruning = False



def read_data(filename):
    f = open(filename, 'r', encoding="utf8")
    all_lines = f.readlines()  # lưu thành 1 string cho mỗi hàng trong file vào list
    labels = all_lines[0].strip().split(',')  # lấy tên thuộc tính ở hàng đầu
    dataset = []
    for line in all_lines[1:]:  # dữ liệu ở hàng thứ 2 trở đi
        line = line.strip().split(',')  # lấy data
        dataset.append(line)
    return dataset, labels


def read_testset(testfile):
    f = open(testfile, 'r', encoding="utf8")
    all_lines = f.readlines()
    testset = []
    for line in all_lines[1:]:
        line = line.strip().split(',')  # lấy data
        testset.append(line)
    return testset

filename = 'dataset.txt'
data, dataLabels = read_data(filename)
del(dataLabels[-1])

# tính entropy
def cal_entropy(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    # tạo từ điển tất cả các lớp
    for featVec in dataset:
        currentlabel = featVec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] += 1
    # tính entropy
    Ent = 0.0
    for key in labelCounts:
        p = float(labelCounts[key]) / numEntries
        Ent = Ent - p * log(p, 2)
    return Ent


# chia dataset theo thuộc tính có index là axis có giá trị value
def splitdataset(dataset, axis, value):
    retdataset = []
    for featVec in dataset:  # lặp với dữ liệu dataset
        if featVec[axis] == value: # nếu giá trị của thuộc tính có index là axis bằng giá trị value thì:
            reducedfeatVec = featVec[:axis]  # copy các giá trị ở bên trái thuộc tính có index là axis vào reducedfeatVec
            reducedfeatVec.extend(featVec[axis + 1:])  # thêm các giá trị ở bên phải thuộc tính có index là axis vào reducedfeatVec
            retdataset.append(reducedfeatVec) # kết quả ra sẽ là dataset không còn thuộc tính có index là axis nữa
    return retdataset


'''
Choosing the best way to divide the data set
ID3 algorithm: select partition attributes based on information gain
C4.5 algorithm: use "gain ratio" to select partition attributes
'''


# Thuật toán ID3
def ID3_chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEnt = cal_entropy(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # lặp cho từng thuộc tính
        featList = [example[i] for example in dataset] # featList là list chứa giá trị của thuộc tính i 
        uniqueVals = set(featList)  # Tạo list chứa giá trị có thể có trong featList
        newEnt = 0.0
        for value in uniqueVals:  # Tính entropy cho mỗi sự lựa chọn chia (test)
            subdataset = splitdataset(dataset, i, value)
            p = len(subdataset) / float(len(dataset))
            newEnt += p * cal_entropy(subdataset)
        infoGain = baseEnt - newEnt
        print(u"Information gain của thuộc tính %d trong ID3 là：%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain  # Lấy ra information gain lớn nhất 
            bestFeature = i
    return bestFeature


# Thuật toán C4.5
def C45_chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEnt = cal_entropy(dataset)
    bestGain_ratio = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # lặp cho từng thuộc tính
        featList = [example[i] for example in dataset]  # featList là list chứa giá trị của thuộc tính i 
        uniqueVals = set(featList)  # Tạo list chứa giá trị có thể có trong featList
        newEnt = 0.0
        IV = 0.0
        for value in uniqueVals:  # Tính entropy cho mỗi sự lựa chọn chia (test)
            subdataset = splitdataset(dataset, i, value)
            p = len(subdataset) / float(len(dataset))
            newEnt += p * cal_entropy(subdataset)
            IV = IV - p * log(p, 2) # thêm so với thuật toán ID3
        infoGain = baseEnt - newEnt
        if (IV == 0):  # không để IV = 0
            continue
        gain_ratio = infoGain / IV  # tính được Gain ratio của thuộc tính đang xét
        print(u"Gain ratio của thuộc tính %d trong ID3 là：%.3f" % (i, gain_ratio))
        if (gain_ratio > bestGain_ratio):
            bestGain_ratio = gain_ratio # Lấy ra Gain ratio lớn nhất 
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    '''
    The data set has processed all attributes, but the class label is still not unique,
    At this point we need to decide how to define the leaf node. In this case, we usually use the majority voting method to determine the classification of the leaf node.
    '''
    classCont = {}
    for vote in classList:
        if vote not in classCont.keys():
            classCont[vote] = 0
        classCont[vote] += 1
    sortedClassCont = sorted(classCont.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCont[0][0]


# Thuật toán ID3 dùng để tạo cây
def ID3_createTree(dataset, labels, test_dataset):
    classList = [example[-1] for example in dataset]  # list chứa giá trị của kết quả trong dataset
    if classList.count(classList[0]) == len(classList):
        # Nếu tất cả các giá trị đều giống nhau, ngừng chia dataset
        return classList[0]
    if len(dataset[0]) == 1:
        # Nếu hết thuộc tính để chia, trả về kết quả xuất hiện nhiều nhất trong dataset
        return majorityCnt(classList)
    bestFeat = ID3_chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    print(u"Tại thời điểm này, thuộc tính dùng để chia là: " + (bestFeatLabel))

    ID3Tree = {bestFeatLabel: {}}  # Tạo cây
    del (labels[bestFeat])  # xoá tên thuộc tính được chọn
    # tạo list chứa những giá trị của thuộc tính được chọn
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)

    if pre_pruning:
        ans = []
        for index in range(len(test_dataset)):
            ans.append(test_dataset[index][-1])  # lấy giá trị kết quả rồi lưu vào list ans
        result_counter = Counter()
        for vec in dataset:
            result_counter[vec[-1]] += 1  # Đếm số lần xuất hiện của mỗi giá trị trong dataset e.g: result_counter = Counter({'one': 9, 'zero': 7})
        leaf_output = result_counter.most_common(1)[0][0]  # Gán tên giá trị có số lần xuất hiện nhiều nhất vào leaf_output. e.g: leaf_output=one
        root_acc = cal_acc(test_output=[leaf_output] * len(test_dataset), label=ans)  # e.g: root_acc = cal_acc(['one', 'one', 'one', 'one', 'one', 'one', 'one'], ['zero', 'one', 'one', 'zero', 'one', 'zero', 'zero'])
        outputs = []
        ans = []
        for value in uniqueVals:
            cut_testset = splitdataset(test_dataset, bestFeat, value)  # chia testdataset theo index của nút chuẩn bị thêm vào cây
            cut_dataset = splitdataset(dataset, bestFeat, value)  # chia dataset theo index của nút chuẩn bị thêm vào cây
            # Mục đích của việc chia này là để tính độ chính xác sau khi chia
            for vec in cut_testset:
                ans.append(vec[-1])  # lấy giá trị kết quả (hàng cuối) lưu vào list ans
            result_counter = Counter() 
            for vec in cut_dataset:
                result_counter[vec[-1]] += 1  # Đếm số lần xuất hiện của mỗi giá trị trong cut_dataset e.g: result_counter = Counter({'zero': 7, 'one': 3})
            leaf_output = result_counter.most_common(1)[0][0]  # Gán tên giá trị có số lần xuất hiện nhiều nhất vào leaf_output. e.g: leaf_output=one
            outputs += [leaf_output] * len(cut_testset)  # Xây dựng test_output
        cut_acc = cal_acc(test_output=outputs, label=ans)   # e.g: root_acc = cal_acc(['zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'one'], ['zero', 'one', 'zero', 'one', 'zero', 'zero', 'one'])

        if cut_acc <= root_acc:  # nếu độ chính xác sau khi chia nhỏ hơn độ chính xác trước khi chia thì cắt tỉa cây
            return leaf_output

    for value in uniqueVals:
        subLabels = labels[:]
        ID3Tree[bestFeatLabel][value] = ID3_createTree(
            splitdataset(dataset, bestFeat, value),
            subLabels,
            splitdataset(test_dataset, bestFeat, value))  # loại bỏ thuộc tính được chọn khỏi dataset, đệ quy hàm cho ID3_createTree cho dataset này

    if post_pruning:
        tree_output = classifytest(ID3Tree,
                                   featLabels=dataLabels,
                                   testDataSet=test_dataset)
        ans = []
        for vec in test_dataset:
            ans.append(vec[-1])  # lấy giá trị kết quả (hàng cuối) của test_dataset
        root_acc = cal_acc(tree_output, ans)  # tính độ chính xác của cây đã chia
        result_counter = Counter()
        for vec in dataset:
            result_counter[vec[-1]] += 1 # Đếm số lần xuất hiện của mỗi giá trị trong dataset
        leaf_output = result_counter.most_common(1)[0][0] # Gán tên giá trị có số lần xuất hiện nhiều nhất vào leaf_output.
        cut_acc = cal_acc([leaf_output] * len(test_dataset), ans) # tính độ chính xác của cây trước khi chia

        if cut_acc >= root_acc: # nếu độ chính xác sau khi chia nhỏ hơn độ chính xác trước khi chia thì cắt tỉa cây
            return leaf_output

    return ID3Tree


def C45_createTree(dataset, labels, test_dataset):
    classList = [example[-1] for example in dataset]  # list chứa giá trị của kết quả trong dataset
    if classList.count(classList[0]) == len(classList):
        # Nếu tất cả các giá trị đều giống nhau, ngừng chia dataset
        return classList[0]
    if len(dataset[0]) == 1:
        # Nếu hết thuộc tính để chia, trả về kết quả xuất hiện nhiều nhất trong dataset
        return majorityCnt(classList)
    bestFeat = C45_chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    print(u"Tại thời điểm này, thuộc tính dùng để chia là: " + (bestFeatLabel))
    C45Tree = {bestFeatLabel: {}}  # Tạo cây
    del (labels[bestFeat])  # xoá tên thuộc tính được chọn
    # tạo list chứa những giá trị của thuộc tính được chọn
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)

    if pre_pruning:
        ans = []
        for index in range(len(test_dataset)):
            ans.append(test_dataset[index][-1])  # lấy giá trị kết quả rồi lưu vào list ans
        result_counter = Counter()
        for vec in dataset:
            result_counter[vec[-1]] += 1  # Đếm số lần xuất hiện của mỗi giá trị trong dataset e.g: result_counter = Counter({'one': 9, 'zero': 7})
        leaf_output = result_counter.most_common(1)[0][0]  # Gán tên giá trị có số lần xuất hiện nhiều nhất vào leaf_output. e.g: leaf_output=one
        root_acc = cal_acc(test_output=[leaf_output] * len(test_dataset), label=ans)  # e.g: root_acc = cal_acc(['one', 'one', 'one', 'one', 'one', 'one', 'one'], ['zero', 'one', 'one', 'zero', 'one', 'zero', 'zero'])
        outputs = []
        ans = []
        for value in uniqueVals:
            cut_testset = splitdataset(test_dataset, bestFeat, value)  # chia testdataset theo index của nút chuẩn bị thêm vào cây
            cut_dataset = splitdataset(dataset, bestFeat, value)  # chia dataset theo index của nút chuẩn bị thêm vào cây
            # Mục đích của việc chia này là để tính độ chính xác sau khi chia
            for vec in cut_testset:
                ans.append(vec[-1])  # lấy giá trị kết quả (hàng cuối) lưu vào list ans
            result_counter = Counter() 
            for vec in cut_dataset:
                result_counter[vec[-1]] += 1  # Đếm số lần xuất hiện của mỗi giá trị trong cut_dataset e.g: result_counter = Counter({'zero': 7, 'one': 3})
            leaf_output = result_counter.most_common(1)[0][0]  # Gán tên giá trị có số lần xuất hiện nhiều nhất vào leaf_output. e.g: leaf_output=one
            outputs += [leaf_output] * len(cut_testset)  # Xây dựng test_output
        cut_acc = cal_acc(test_output=outputs, label=ans)   # e.g: root_acc = cal_acc(['zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'one'], ['zero', 'one', 'zero', 'one', 'zero', 'zero', 'one'])

        if cut_acc <= root_acc:  # nếu độ chính xác sau khi chia nhỏ hơn độ chính xác trước khi chia thì cắt tỉa cây
            return leaf_output

    for value in uniqueVals:
        subLabels = labels[:]
        C45Tree[bestFeatLabel][value] = C45_createTree(
            splitdataset(dataset, bestFeat, value),
            subLabels,
            splitdataset(test_dataset, bestFeat, value))  # loại bỏ thuộc tính được chọn khỏi dataset, đệ quy hàm cho ID3_createTree cho dataset này

    if post_pruning:
        tree_output = classifytest(C45Tree,
                                   featLabels=dataLabels,
                                   testDataSet=test_dataset)
        ans = []
        for vec in test_dataset:
            ans.append(vec[-1])  # lấy giá trị kết quả (hàng cuối) của test_dataset
        root_acc = cal_acc(tree_output, ans)  # tính độ chính xác của cây đã chia
        result_counter = Counter()
        for vec in dataset:
            result_counter[vec[-1]] += 1 # Đếm số lần xuất hiện của mỗi giá trị trong dataset
        leaf_output = result_counter.most_common(1)[0][0] # Gán tên giá trị có số lần xuất hiện nhiều nhất vào leaf_output.
        cut_acc = cal_acc([leaf_output] * len(test_dataset), ans) # tính độ chính xác của cây trước khi chia

        if cut_acc >= root_acc: # nếu độ chính xác sau khi chia nhỏ hơn độ chính xác trước khi chia thì cắt tỉa cây
            return leaf_output

    return C45Tree


def classify(inputTree, featLabels, testVec):
    """
    Input: decision tree, classification label, test data
    Output: decision result
    Description: Run decision tree
    """
    firstStr = list(inputTree.keys())[0] # lấy gốc e.g: firstStr = 'have a job'
    secondDict = inputTree[firstStr] # lấy sự lựa chọn của gốc e.g: {'zero': 'zero', 'one': 'one'}
    featIndex = featLabels.index(firstStr) # lấy index của firstStr
    classLabel = '0'
    for key in secondDict.keys(): # lặp cho từng sự lựa chọn
        if testVec[featIndex] == key: # nêu hàng trong testDataSet trùng với sự lựa chọn đang xét
            if type(secondDict[key]).__name__ == 'dict': 
                classLabel = classify(secondDict[key], featLabels, testVec) # nếu cây đang xét có nhánh thì đệ quy cho đến khi xét được lá
            else:
                classLabel = secondDict[key] # classLabel = lá mà có hàng trong testDataSet trùng với sự lựa chọn đang xét
    return classLabel


def classifytest(inputTree, featLabels, testDataSet):
    """
    Input: decision tree, classification label, test data
    Output: decision result
    Description: Run decision tree
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec)) # lưu tất cả những label lấy được từ classify()
    return classLabelAll


def cal_acc(test_output, label):
    """
    :param test_output: the output of testset
    :param label: the answer
    :return: the acc of
    """
    assert len(test_output) == len(label)
    count = 0
    for index in range(len(test_output)):  # lặp cho từng giá trị trong test_output
        if test_output[index] == label[index]:
            count += 1  # đếm số lượng giá trị trùng nhau của test_output và label trong cùng 1 index

    return float(count / len(test_output))  # trả về giá trị độ chính xác


if __name__ == '__main__':
    filename = 'dataset.txt'
    testfile = 'testset.txt'
    dataset, labels = read_data(filename)
    # dataset,features=createDataSet()
    print('Dữ liệu: ', dataset)
    print("---------------------------------------------")
    print(u"Số hàng dữ liệu: ", len(dataset))
    print("Entropy của dữ liệu: ", cal_entropy(dataset))
    print("---------------------------------------------")

    while True:
        dec_tree = '2'
        # ID3 decision tree
        if dec_tree == '1':
            print(u"Thuộc tính dùng để chia trong thuật toán ID3 là: " + str(ID3_chooseBestFeatureToSplit(dataset)))
            print("--------------------------------------------------")
            labels_tmp = labels[:]  # Copy, createTree will change the labels
            ID3desicionTree = ID3_createTree(dataset, labels_tmp, test_dataset=read_testset(testfile))
            print('Cây quyết định ID3: \n', ID3desicionTree)
            # treePlotter.createPlot(ID3desicionTree)
            treePlotter.ID3_Tree(ID3desicionTree)
            testSet = read_testset(testfile)
            print("Kết quả của test dataset: ")
            print('ID3_TestSet_classifyResult:\n', classifytest(ID3desicionTree, labels, testSet))
            print("---------------------------------------------")

        # C4.5 decision tree
        if dec_tree == '2':
            print(u"Thuộc tính đầu tiên dùng để chia trong thuật toán C4.5 là: " + labels[C45_chooseBestFeatureToSplit(dataset)])
            print("--------------------------------------------------")
            labels_tmp = labels[:]  # Copy, createTree will change the labels
            C45desicionTree = C45_createTree(dataset, labels_tmp, test_dataset=read_testset(testfile))
            print('Cây quyết định C4.5: \n', C45desicionTree)
            treePlotter.C45_Tree(C45desicionTree)
            testSet = read_testset(testfile)
            print("Kết quả của test dataset: ")
            print('C4.5_TestSet_classifyResult:\n', classifytest(C45desicionTree, labels, testSet))
            print("---------------------------------------------")

        break
