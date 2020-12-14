import os                                                                                                                                                                                                             
from bs4 import BeautifulSoup 
from io import open

def main():
    # gets the directory of the data folder
    dir = os.getcwd()
    dir = dir + "\data"
    print(dir)

    # appends the name of each xml file to list r
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
    # iterates through the list of all filenames
    for file in r:
        # gets the relative file path of the file
        filepath = file.split("\data",1)[1] 

        # opens file using filepath
        with open("data" + filepath, 'r', encoding="utf-8") as f: 
            open_file = f.read() 
        data = BeautifulSoup(open_file, "xml") 

        # gets all paragraph and headline tags in xml file
        paragraph = data.find_all('p')
        paragraph += data.find_all('hl1')
        relevent_p = []
        for x in paragraph:
            text = x.text.lower()
            # checks if 'bush' and 'gore' are in the text, if so, it appends it to relevant_p list
            if 'bush' in text or 'gore' in text:
                relevent_p.append(text)
        if relevent_p:
            # appends the publication date attributes to output
            output = data.find('meta', {'name': 'publication_year'}).get('content') + " "
            output += data.find('meta', {'name': 'publication_month'}).get('content') + " "
            output += data.find('meta', {'name': 'publication_day_of_month'}).get('content') + "\t"
            for x in relevent_p:
                output += x + " "
            output += "\n"
            # append publication date information to dataArr list
            dataArr.append(output)

    print('length of data array: ')
    print(len(dataArr))

    # writes dataArr list components to a file named "articles.txt"
    file1 = open("articles.txt","w", encoding="utf-8") 
    file1.writelines(dataArr) 
    file1.close() 

if __name__ == "__main__":
    main()
