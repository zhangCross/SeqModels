# SeqModels
some sequence models

Each directory is a separate sequence model.

1. cd generate_university_name. generate university names by rnn model.  
university.txt is trainning courpus file.   
python rnn.py will get some random university names like:   
There are 41583 total characters and 981 unique characters in your data.
55.0207418171941
Iteration: 0, Loss: 55.041496
Iteration: 5000, Loss: 29.220552
Iteration: 10000, Loss: 25.493005
Iteration: 15000, Loss: 23.411126
Iteration: 20000, Loss: 21.902653
Iteration: 25000, Loss: 20.932928
final loss: 20.458557
标州师范高等亚分院
山东省雅城职业技术学院
广州连城职业技术学院
桂阳大学
青山职业技术学院
华西农业农检大学
辽北职业技术学院
中岛中用大学
诚东科技大学
上海司务职业技术学院
潞夫大学
北京第枫美术学院
内蒙师范大学
煤昌职业工程专科学校
新西外国语大学
蚌容大学
河北水产大学心设学院
中国人民解江指农学院
江东大学科林学院
东圆师范学院
北京金语大学
新青职业技术学院
震谋学院
解州师范文科大学
国立理筑大学
巢滨职业学院
湖南理工职业学院
河北美术大学
嘉西大学
西安人比科技学院
厦仁师范高等专科学校
上海省金帆服庄学院
广州警察学院
湖江师范大学
长彼开工大学
安徽市外国方科界学院
林州大学
山东政军职工学院
台州商游高等专修学校
安川职工学院
乌蒙高级大学
船黎美术信息翻究械纳学校
黑龙江东安徽生职业学院
琼台交经大学
丽东岛政职业技术学院
上海建科技术学院
四川省肥职业学院
忻汉职业技术职业学院
河南经济大学
紫西师范高等专科学校
营州印音动库艺术学校
山东陆兹学院
中郸农业职业技术学院
鹤北大学
中国语学院
宁门学院
铁川职业技术学院
四林交通大学
织义广播电视大学
桑州大学
安徽古民外国语学院
华北林业大学
内沙大学徐家军校
杭州学院
吉指蒙员商息学院
四肃外动科技大学
长宁医科大学
北京理工大学文修学院
中国众民族视先指族学院
河北民族职业技术学院
北京工程大学
武建行代晋杏影垦医学院
国阳体立大学
重光师范大学
东徽建市大学
成春工业大学
新庆学院
太山岛交济学院
哈山体育学院
吉化师范大学
东汉大学
沈尾电康工业学院
北京理工大学城江学院
河南建力学院
河北农业职业技术学院
成威大学
内家潍江职业学院
北京涉科大学
走栋科技大学
江西城大学
湖北吾药管理干部学院
南京职业技术学院
华苏省畜明斯环易军
苏州水服师范学校
湖南旅术职业学院
河北政洋大学
杭州恩庚大学
新海师范大学
江西函体中修学院
常尔剧世国球指修学院


2. cd imdb_sentiment_classification,该项目对tf自带数据imdb进行情感分类，画出训练过程中的准确率和loss，返回最终loss和准确率.   
首先需要解压glove.50d.txt.zip文件得到glove.50d.txt   
使用avg池化层：python avg_pooling.py   
使用lstm层且使用外部的embedding向量实现迁移学习：python lstm.py
