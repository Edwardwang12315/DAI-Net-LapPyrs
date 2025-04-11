#--coding:utf-8--
import torch
from collections import OrderedDict

def load_and_modify_vgg_weights(model_type, weight_path1,weight_path2,weight_path3, new_weight_path):
    """
    读取并修改VGG13或VGG16的权重文件中的键名。

    参数:
        model_type (str): 模型类型，可以是'vgg13'或'vgg16'。
        weight_path (str): 权重文件的路径。
        new_weight_path (str): 保存修改后权重文件的路径。
    """
    # 加载权重文件
    weights1 = torch.load(weight_path1)
    weights2 = torch.load( weight_path2 )
    weights3 = torch.load(weight_path3)
    # 创建一个新的OrderedDict来存储修改后的权重
    new_weights = OrderedDict()

    # exit()
    # 遍历原始权重字典，修改键名
    for key, value in weights2.items():
        # if key!='classifier.0.weight' and key!='classifier.0.bias' and key!='classifier.3.weight' and key!='classifier.3.bias'and key!='classifier.6.weight' and key!='classifier.6.bias':
        if model_type == 'vgg13':
            new_key=key.replace('features.', '')

        elif model_type == 'vgg16':
            if key == '0.weight':
                new_key = key.replace('0.weight', '0.weight')
            elif key == '0.bias':
                new_key = key.replace('0.bias', '0.bias')
            elif key == '2.weight':
                new_key = key.replace('2.weight', '3.weight')
            elif key == '2.bias':
                new_key = key.replace('2.bias', '3.bias')
            elif key == '5.weight':
                new_key = key.replace('5.weight', '7.weight')
            elif key == '5.bias':
                new_key = key.replace('5.bias', '7.bias')
            elif key == '7.weight':
                new_key = key.replace('7.weight', '10.weight')
            elif key == '7.bias':
                new_key = key.replace('7.bias', '10.bias')
            elif key == '10.weight':
                new_key = key.replace('10.weight', '14.weight')
            elif key == '10.bias':
                new_key = key.replace('10.bias', '14.bias')
            elif key == '12.weight':
                new_key = key.replace('12.weight', '17.weight')
            elif key == '12.bias':
                new_key = key.replace('12.bias', '17.bias')
            elif key == '14.weight':
                new_key = key.replace('14.weight', '20.weight')
            elif key == '14.bias':
                new_key = key.replace('14.bias', '29.bias')
            elif key == '17.weight':
                new_key = key.replace('17.weight', '24.weight')
            elif key == '17.bias':
                new_key = key.replace('17.bias', '24.bias')
            elif key == '19.weight':
                new_key = key.replace('19.weight', '27.weight')
            elif key == '19.bias':
                new_key = key.replace('19.bias', '27.bias')
            elif key == '21.weight':
                new_key = key.replace('21.weight', '30.weight')
            elif key == '21.bias':
                new_key = key.replace('21.bias', '30.bias')
            elif key == '24.weight':
                new_key = key.replace('24.weight', '34.weight')
            elif key == '24.bias':
                new_key = key.replace('24.bias', '34.bias')
            elif key == '26.weight':
                new_key = key.replace('26.weight', '37.weight')
            elif key == '26.bias':
                new_key = key.replace('26.bias', '37.bias')
            elif key == '28.weight':
                new_key = key.replace('28.weight', '40.weight')
            elif key == '28.bias':
                new_key = key.replace('28.bias', '40.bias')
            elif key == '31.weight':
                new_key = key.replace('31.weight', '44.weight')
            elif key == '31.bias':
                new_key = key.replace('31.bias', '44.bias')
            elif key == '33.weight':
                new_key = key.replace('33.weight', '47.weight')
            elif key == '33.bias':
                new_key = key.replace('33.bias', '47.bias')
                
            new_weights[new_key] = value
    
    # 保存修改后的权重文件
    torch.save(new_weights, new_weight_path)

# 示例使用
load_and_modify_vgg_weights('vgg16',weight_path1 ='./vgg16_reducedfc.pth',weight_path2='./vgg16_reducedfc.pth', weight_path3 = './vgg16_reducedfc.pth',new_weight_path = './vgg16_withBN.pth')
