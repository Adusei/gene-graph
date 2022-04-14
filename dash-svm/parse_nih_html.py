import requests
from bs4 import BeautifulSoup


def main(pubmed_id):

    url = 'https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/'.format(pubmed_id=pubmed_id)
    r = requests.get(url)

    soup = BeautifulSoup(r.content, 'html.parser')
    title = soup.title.text
    abstract = soup.find("div", {"id": "enc-abstract"}).text

    return title, abstract, url


if __name__ == '__main__':
    title, abstract, url = main()
    print(title)
    print(abstract)
    print(url)
