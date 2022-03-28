from bs4 import BeautifulSoup

def read_from_xml():

    # Reading the data inside the xml
    # file to a variable under the name
    # data
    with open('read_from_xml/pic_1135.xml', 'r') as f:
        data = f.read()

    # Passing the stored data inside
    # the beautifulsoup parser, storing
    # the returned object
    Bs_data = BeautifulSoup(data, "xml")
    # p = Bs_data.findall('<P((.|\s)+?)</P>', str(response))
    # Bs_data.find_all('document')[1].extract()
    # Finding all instances of tag
    # `unique`
    b_unique = Bs_data.find_all('xmin')
    x=b_unique.pop(0)
    print(type(b_unique))
    # print(x)
    # str1=str(b_unique)
    # print(str1[0])
    print(type(x))
    str='abc'
    print(str[0])
    print(str[2])


    # # Using find() to extract attributes
    # # of the first instance of the tag
    # b_name = Bs_data.find('child', {'name': 'Frank'})
    #
    # print(b_name)
    #
    # # Extracting the data stored in a
    # # specific attribute of the
    # # `child` tag
    # value = b_name.get('test')
    #
    # print(value)
read_from_xml()
