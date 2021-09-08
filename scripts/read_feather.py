import pandas as pd
import requests
import io

from lxml import etree


BASE_URL = 'https://autumn-data.s3-ap-southeast-2.amazonaws.com/covid_19/philippines/1608531748/454a2c4/data/full_model_runs/'
CHAINS = ["chain-"+str(i) for i in range (0,7)]
FILETYPE = ["derived_outputs.feather", "mcmc_param.feather", "mcmc_run.feather", "outputs.feather"]


URL = []
DF = []

for each in FILETYPE:
    for chain in CHAINS:
        URL.append(BASE_URL+chain+"/"+each)

for each in URL[:2]:
    DF.append(pd.read_feather(each))


DF = pd.concat(DF, ignore_index= True)
DF




pd.read_feather("https://autumn-data.s3-ap-southeast-2.amazonaws.com/covid_19/philippines/1608531748/454a2c4/data/full_model_runs/chain-0/derived_outputs.feather")

URL[6]



res = requests.get(
    BASE_URL,
    stream=True
)

ltree = etree.parse(BASE_URL)

root = ltree.getroot()

a_dict = dict()

for element in root.iter("script"):
    element.text

a_dict

a_dict.find('true')
a_dict.replace('true', '"true"')
y = exec(a_dict)

for el in ltree.xpath('descendant-or-self::text()'):
    print (el)

a_dict = exec(list(ltree.xpath('descendant-or-self::text()'))[247])

type(ltree)
dir(ltree)




for child in root[1]:
    print (child.attrib)

root[0][3]
root[0][3]



root.tag
print(etree.tostring(root, pretty_print=True))


root[1].attrib

root.findall(".//link[@url]")[0].tag

res.raw.decode_content = True
mem_fh = io.BytesIO(res.raw.read())
derived_output_df = pd.read_feather(mem_fh)

derived_output_df.columns
derived_output_df


page = requests.get('http://www.autumn-data.com/app/covid_19/region/philippines/run/1608531748-454a2c4.html')
tree = html.fromstring(page.content)

y = html.parse('http://www.autumn-data.com/app/covid_19/region/philippines/run/1608531748-454a2c4.html')
y.parse

res = requests.get('http://www.autumn-data.com/app/covid_19/region/philippines/run/1608531748-454a2c4.html')


name = doc.xpath("\\")   
print (name)



import requests 
import lxml.html 

# requesting url 
web_response = requests.get('http://www.autumn-data.com/app/covid_19/region/philippines/run/1608531748-454a2c4.html') 

# building 
element_tree = lxml.html.fromstring(web_response.text) 

tree_title_element = element_tree.xpath('//title')[0] 

print("Tag title : ", tree_title_element.tag) 
print("\nText title :", tree_title_element.text_content()) 
print("\nhtml title :", lxml.html.tostring(tree_title_element)) 
print("\ntitle tag:", tree_title_element.tag) 
print("\nParent's tag title:", tree_title_element.getparent().tag) 

chains = ["chain-"+str(each) for each in range(0,6)]
filetypes = ["derived_outputs","mcmc_params","mcmc_run","outputs"]
filetypes = [each+".feather" for each in filetypes]
filetypes = filetypes[0]


x = pd.read_feather("C:/Users/maba0001/AuTuMN/data/outputs/run/covid_19/sri_lanka/2021-07-06--11-19-56/outputs/derived_outputs.feather")

x = list(x.columns)
x.sort()
x.vaccinationXagegroup_40