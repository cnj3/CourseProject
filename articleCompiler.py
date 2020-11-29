#pip install bs4
#pip install lmxml

import os                                                                                                                                                                                                             
from bs4 import BeautifulSoup 


def main():
    dir = os.getcwd()
    dir = dir + "\data"
    print(dir)
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = next(os.walk(subdir))[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:   
                if file.endswith(".xml"):                                                                                 
                    r.append(os.path.join(subdir, file))                                                                         
    print(len(r)) 
    dataArr = []
    for file in r:
        filepath = file.split("\data",1)[1] 
        with open("data" + filepath, 'r', encoding="utf-8") as f: 
            open_file = f.read() 
        data = BeautifulSoup(open_file, "xml") 

        paragraph = data.find_all('p')
        relevent_p = []
        for x in paragraph:
            text = x.text.lower()
            if 'bush' in text or 'gore' in text:
                relevent_p.append(text)
        if relevent_p:
            output =  data.find('meta', {'name': 'publication_year'}).get('content') + " "
            output += data.find('meta', {'name': 'publication_month'}).get('content') + " "
            output += data.find('meta', {'name': 'publication_day_of_month'}).get('content') + "\t"
            for x in relevent_p:
                output += x + " "
            output += "\n"
            dataArr.append(output)

    print('length of data array: ')
    print(len(dataArr))

    file1 = open("articles.txt","w", encoding="utf-8") 
    file1.writelines(dataArr) 
    file1.close() 

if __name__ == "__main__":
    main()