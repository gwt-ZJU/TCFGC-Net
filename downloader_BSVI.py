import urllib.request  #打开网页模块
import urllib.parse    #转码模块
import os
import requests

#这里的路径可替换为自己保存文件夹的路径
save_path = r"F:\202006BaiDu\picture_save/"

ak = "填入你的ak码"

#判断文件夹是否存在，若不存在则创建
if not os.path.exists(save_path):
    os.makedirs(save_path)

#替换为你自己制作的txt路径及文件
#从txt文件中读取坐标
with open(r"F:\202006BaiDu\location.txt","r",encoding='UTF-8')as f:
    location = f.readlines()

#使用for循环遍历出每个location坐标
for i in range(len(location)):
    #使用for循环，每一次都输出[0,1,2,3]这一列表
    for j in range(4):

        # 将列表中的第i个拿出来，并用split划分拿第0个
        location_number = (location[i].split(';')[0]).replace("\n", "")
        location_number = location_number.replace(";", "")

        #旋转的角度
        #[0,1,2,3] * 90 = [0,90,180,270]
        heading_number = str(90*j)

        url = r"https://api.map.baidu.com/panorama/v2?" \
              "&width=1024&height=512" \
              "&location="+location_number+\
              "&heading="+heading_number+ \
              "&ak=" + ak

        #文件保存名称
        save_name =save_path+str(i)+"."+str(j)+".jpg"
        print(url)
        #打开网页
        rep = urllib.request.urlopen(url)
        #将图片存入本地，创建一个save_name的文件，wb为写入
        f = open(save_name,'wb')
        #写入图片
        f.write(rep.read())
        f.close()
        print('图片保存成功')