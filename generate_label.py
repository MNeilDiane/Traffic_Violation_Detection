import os

train_txt_path = os.path.join("data", "catVSdog", "train.txt")
train_dir = os.path.join("data", "catVSdog", "train_data")
test_txt_path = os.path.join("data", "catVSdog", "test.txt")
test_dir = os.path.join("data", "catVSdog", "test_data")

def generate_txt(txt_path, img_dir):
    f = open(txt_path, 'w')

    # 返回的是一个三元组(root,dirs,files)
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    # topdown 为True 优先遍历root文件夹，false优先遍历root的子目录
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)
            #获取root文件夹子文件夹的绝对路径
            img_list = os.listdir(i_dir)
            for i in range(len(img_list)):
                if not img_list[i].endwith('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'):
                    continue
                label = img_list[i].split('.')[0]
                if label == 'cat':
                    label ='0'
                else:
                    label = '1'

    f.close()

if __name__ == '__main__':
    generate_txt(train_txt_path, train_dir)
    generate_txt(test_txt_path, test_dir)






