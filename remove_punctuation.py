import re


def main():
    with open('data/articles.txt') as articles_txt:
        lines = articles_txt.readlines()

    with open('data/edited_articles.txt', 'w+') as new_articles_txt:
        for line in lines:
            line = re.sub('[^\w\s]', '', line)
            new_articles_txt.write(line)


if __name__ == '__main__':
    main()
