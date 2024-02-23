# -*- coding: utf-8 -*-
import xmltodict
import os
import sys
import json
import io
import glob
from xml.dom.minidom import Document
global null
import argparse


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                L.append(os.path.join(root, file))
        return L
 
def main(path, destPath):
    path_list = file_name(path)    
    for name in enumerate(path_list):
        m_path = name[1]               
        dir = os.path.dirname(m_path)     
        file_json = io.open(m_path, 'r', encoding='utf-8')
        json_data = file_json.read()                
        data = json.loads(json_data)
        m_filename = data['metadata']['name']
        m_filename = m_filename.replace(" ","")
        m_path = os.path.join(dir,m_filename)
        m_width = data['metadata']['width']
        m_height = data['metadata']['height']
        object_name = os.path.splitext(m_filename)[0]
        new_object_name = object_name + '.xml'
        #print(new_object_name)
        doc = Document()  # 创建DOM文档对象
        DOCUMENT = doc.createElement('annotation')  # 创建根元素
     
        folder= doc.createElement('folder')
        folder_text = doc.createTextNode(m_folder)
        folder.appendChild(folder_text)
        DOCUMENT.appendChild(folder)
        doc.appendChild(DOCUMENT)
     
        filename = doc.createElement('filename')
        filename_text = doc.createTextNode(m_filename)
        filename.appendChild(filename_text)
        DOCUMENT.appendChild(filename)
        doc.appendChild(DOCUMENT)
     
        path = doc.createElement('path')
        path_text = doc.createTextNode(m_filename)
        path.appendChild(path_text)
        DOCUMENT.appendChild(path)
        doc.appendChild(DOCUMENT)
     
        source = doc.createElement('source')
        database = doc.createElement('database')
        database_text = doc.createTextNode(m_database)  # 元素内容写入
        database.appendChild(database_text)
        source.appendChild(database)
        DOCUMENT.appendChild(source)
        doc.appendChild(DOCUMENT)
     
        size = doc.createElement('size')
        width = doc.createElement('width')
        width_text = doc.createTextNode(str(m_width))  # 元素内容写入
        width.appendChild(width_text)
        size.appendChild(width)
     
        height = doc.createElement('height')
        height_text = doc.createTextNode(str(m_height))
        height.appendChild(height_text)
        size.appendChild(height)
     
        depth = doc.createElement('depth')
        depth_text = doc.createTextNode(str(m_depth))
        depth.appendChild(depth_text)
        size.appendChild(depth)
     
        DOCUMENT.appendChild(size)
     
        segmented = doc.createElement('segmented')
        segmented_text = doc.createTextNode(str(m_segmented))
        segmented.appendChild(segmented_text)
        DOCUMENT.appendChild(segmented)
        doc.appendChild(DOCUMENT)
        value = data.get('instances')       
        if (value):
            for i in range(len(data['instances'])):
                m_xmin_0 = data['instances'][i]['left']
                m_ymin_0 = data['instances'][i]['top']
                m_xmax_0 = data['instances'][i]['left'] + data['instances'][i]['width']
                m_ymax_0 = data['instances'][i]['top'] + data['instances'][i]['height']
               
                for j in range (len(data['instances'][i]['attributes'])):
                    m_name_0 = data['instances'][i]['attributes'][j]['class_id']
                
                object = doc.createElement('object')
                name = doc.createElement('name')
                name_text = doc.createTextNode(str(m_name_0))
                name.appendChild(name_text)
                object.appendChild(name)
         
                pose = doc.createElement('pose')
                pose_text = doc.createTextNode(m_pose)
                pose.appendChild(pose_text)
                object.appendChild(pose)
         
                truncated = doc.createElement('truncated')
                truncated_text = doc.createTextNode(str(m_truncated))
                truncated.appendChild(truncated_text)
                object.appendChild(truncated)
         
                difficult = doc.createElement('difficult')
                difficult_text = doc.createTextNode(str(m_difficult))
                difficult.appendChild(difficult_text)
                object.appendChild(difficult)
         
                bndbox = doc.createElement('bndbox')
                xmin = doc.createElement('xmin')
                xmin_text = doc.createTextNode(str(int(m_xmin_0)))
                xmin.appendChild(xmin_text)
                bndbox.appendChild(xmin)
         
                ymin = doc.createElement('ymin')
                ymin_text = doc.createTextNode(str(int(m_ymin_0)))
                ymin.appendChild(ymin_text)
                bndbox.appendChild(ymin)
         
                xmax = doc.createElement('xmax')
                xmax_text = doc.createTextNode(str(int(m_xmax_0)))
                xmax.appendChild(xmax_text)
                bndbox.appendChild(xmax)
         
                ymax = doc.createElement('ymax')
                ymax_text = doc.createTextNode(str(int(m_ymax_0)))
                ymax.appendChild(ymax_text)
                bndbox.appendChild(ymax)
                object.appendChild(bndbox)
         
                DOCUMENT.appendChild(object)
                new_path_filename = os.path.join(destPath ,new_object_name)
                #print('new_path_filename=', new_path_filename)
                f = open(new_path_filename, 'w')    
                doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
                f.close()                 

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path",help="input folder",default='input folder with json/images')   
    ap.add_argument("destPath",help="pth file path",default='output folder with pts/images')
    args = ap.parse_args()
    path = args.path
    destPath = args.destPath
    print(path)
    print(destPath)
    null = ''
    #m_folder = os.path.basename(path)
    m_folder = 'VOC2007'
    m_database = 'Unknown'
    m_depth = 3
    m_segmented = 0
    m_pose = 'Unspecified'
    m_truncated = 0
    m_difficult = 0
    m_segmented = 0
    main(args.path, args.destPath)
    
    json_list = file_name(path)
    for name in enumerate(json_list):
        json_path = name[1]
        os.remove(json_path)
    
    