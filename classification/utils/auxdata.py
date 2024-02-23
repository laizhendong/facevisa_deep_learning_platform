import os,json
def save_aux_data(aux_path,image_tags, instances,width, height):
    aux_data = {
        "metadata": {"width":width, "height":height},
    }
    if len(image_tags) > 0:
        aux_data["image_tags"] = []    
        for image_tag in image_tags:
            score_value = 1.0
            if "score" in image_tag.keys():
                score_value = image_tag["score"]
            aux_data['image_tags'].append(
                { "class_id": image_tag['class_id'],"class_name": image_tag['class_name'], "score":score_value }
                )
    if len(instances) > 0:
        aux_data['instances'] = []
        for inst in instances:
            score_value = 1.0
            if "score" in inst.keys():
                score_value = inst['score']
            aux_data['instances'].append(
                {
                    "type":"Rect","left":inst['xywh'][0],"top":inst['xywh'][1],"width":inst['xywh'][2],"height":inst['xywh'][3],
                    "attributes":[
                        {"class_id":inst['class_id'],"class_name":inst['class_name'], "score":score_value}
                    ],
                    "id":inst['id']
                } 
            ) 
    with open(aux_path,'w', encoding="utf-8") as f:
        json.dump(aux_data,f,indent=4)
    return
        
def load_aux_data(image_path):
    aux_file = image_path + ".json"
    if not os.path.exists(aux_file):
        return [],[]
    image_tags, instances = [], []
    with open(aux_file,'r',encoding='utf-8') as f:
        aux_data = json.load(f)
    if 'image_tags' in aux_data.keys():
        for item in aux_data['image_tags']:
            score_value = 1.0
            if "score" in item.keys():
                score_value = item['score']
            image_tags.append(
                {"class_id":item['class_id'], "class_name":item['class_name'], "score": score_value}
            )
    if 'instances' in aux_data.keys():
        for item in aux_data['instances']:
            if item['type'].lower() != 'rect':
                print("!!!unk type: {}".format(item['type']))
                continue
            score_value = 1.0
            if "score" in item.keys():
                score_value = item['score']
            instances.append(
                {
                    "id": "" if "id" not in item.keys() else item['id'],
                    "class_id" : item['attributes'][0]['class_id'],
                    "class_name" : item['attributes'][0]['class_name'],
                    "score" : score_value,
                    "xywh":[ int(float(item['left'])),int(float(item['top'])), int(float(item['width'])), int(float(item['height']))  ]
                }
            )
    return image_tags, instances
         
        
