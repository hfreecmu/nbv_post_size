#!/usr/bin/env python3

import os
import xml.etree.ElementTree as ET

skip_model_ids = ['sun',
                  'dirt',
                  'Apriltag',
                  'tree']

def search_single(node, tag_id, throw_error, sdf_path):
    res = None
    for child in node:
        if child.tag == tag_id:
            if res is not None:
                if throw_error:
                    raise RuntimeError(tag_id + ' found twice in ' + sdf_path)
                else:
                    return None
            else:
                res = child
    
    if res is None:
        if throw_error:
            raise RuntimeError(tag_id + ' not found in ' + sdf_path)
        else:
            return None

    return res

def parse_link(link, sdf_path, model_prefixes, link_dict):
    #right now we are just getting name and scale info
    #we can add other details later
    link_name = link.attrib['name']
    
    visual = search_single(link, 'visual', True, sdf_path)
    geometry = search_single(visual, 'geometry', True, sdf_path)
    mesh = search_single(geometry, 'mesh', False, sdf_path)
    if mesh is None:
        return
    scale = search_single(mesh, 'scale', True, sdf_path)

    model_prefixes.append(link_name)
    link_id = '::'.join(model_prefixes)

    if link_dict.get(link_id) is not None:
        raise RuntimeError('link_id appears twice: ' + link_id)
    
    scale = scale.text.split(' ')
    if len(scale) != 3:
        raise RuntimeError('scale len not 3 for ' + link_id + ' in ' + sdf_path)

    for i in range(len(scale)):
        scale[i] = float(scale[i])

    link_dict[link_id] = {'scale': scale}
        
def parse_model(model, sdf_path, gazebo_model_path, model_prefixes, model_name, link_dict):
    if (model_name is None):
        model_name = model.attrib['name']

    model_prefixes.append(model_name)

    for child in model:
        if child.tag == 'model':
            raise RuntimeError('I know I wrote to support this but why is it here?')
            parse_model(child, sdf_path, gazebo_model_path, model_prefixes.copy(), None, link_dict)
        elif child.tag == 'link':
            parse_link(child, sdf_path, model_prefixes.copy(), link_dict)
        elif child.tag == 'include':
            continue
            raise RuntimeError('I know I wrote to support this but why is it here?')
            parse_include(child, sdf_path, gazebo_model_path, model_prefixes.copy(), link_dict)
        else:
            continue

def parse_sdf(sdf_path, gazebo_model_path, model_prefixes, model_name, link_dict):
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    model = search_single(root, 'model', False, sdf_path)

    if model != None:
        parse_model(model, sdf_path, gazebo_model_path, model_prefixes, model_name, link_dict)
