import ndjson

def genereate_cities_dataset():
    greek_towns = []

    for town_type in ("city", "town", "village"):
        with open(f'gr/place-{town_type}.ndjson', 'r', encoding='utf-8') as f:
            towns = ndjson.load(f)
        for record in towns:
            #print(record.keys())
            try:
                greek_towns.append(preprocess_town_names(record["other_names"]["name:en"]))
                #print(record["other_names"]["name:en"].replace(' ','-'))
            except: 
                try:
                    greek_towns.append(preprocess_town_names(record["other_names"]["int_name"]))
                    #print(record["other_names"]["name:en"].replace(' ','-'))
                except: pass
    
    #Remove duplicates and assert that only english characters appear + '-' --> 26 (no 'q' in greek towns)
    greek_towns = list(set(greek_towns))
    chars = sorted(list(set(''.join(greek_towns))))
    assert(len(chars) == 26)


    # Write city/town/village names to file
    with open('greek_towns.txt', 'w', encoding='utf-8') as f:
        for item in greek_towns:
            f.write(f"{item}\n")

    print("Greek towns dataset generated!")

    return

def preprocess_town_names(town):

    town = town.split(',')[0]
    town = town.split('(')[0]
    if town[-1] == " ":
        town = town[:-1]

    # Transform to lowercase and replace spaces with dashes
    town = town.lower().replace(' ','-')

    # Transform faulty non-english characters to english
    town = town.replace('.','ios')
    town = town.replace('á','a')
    town = town.replace('é','e')
    town = town.replace('í','i')
    town = town.replace('ï','i')
    town = town.replace('ó','o')
    town = town.replace('ú','u')
    town = town.replace('̱','')
    town = town.replace('α','a') 
    town = town.replace('εγρηγόρος','egrigoros')
    town = town.replace('ο','o')
    town = town.replace('ρ','r')
    town = town.replace('τ','t')
    town = town.replace('о','o')
    town = town.replace('ḯ','i')
    town = town.replace('а','a') 

    #Replace translated words to greek with english characters
    town = town.replace('saint','agios') #
    town = town.replace('ancient-kleonai','archaies-kleones')
    town = town.replace('ancient','archaia')
    town = town.replace('new','nea')
    town = town.replace('old','palies') 
    town = town.replace('agii','agioi')

    return town

if __name__ == "__main__":
    genereate_cities_dataset()