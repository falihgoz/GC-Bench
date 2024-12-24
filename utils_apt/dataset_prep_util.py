####################

# references:
# [1] https://github.com/DART-Laboratory/Flash-IDS
# [2] Our contribution

####################

import json
import pandas as pd

from utils_apt.dataset_constants import SupportedDataset, raise_unsupported_dataset

####################
# ref. [1]
def _get_infos_theia(data: list) -> list:
    info = []
    for x in data:
        try:
            action = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['type']
        except:
            action = ''
        try:
            actor = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            actor = ''
        try:
            obj = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject']['com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            obj = ''
        try:
            timestamp = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['timestampNanos']
        except:
            timestamp = ''
        try:
            cmd = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['properties']['map']['cmdLine']
        except:
            cmd = ''
        try:
            path = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObjectPath']['string']
        except:
            path = ''
        try:
            path2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2Path']['string']
        except:
            path2 = ''
        try:
            obj2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2']['com.bbn.tc.schema.avro.cdm18.UUID']
            info.append({'actorID':actor,'objectID':obj2,'action':action,'timestamp':timestamp,'exec':cmd, 'path':path2})
        except:
            pass
        
        info.append({'actorID':actor,'objectID':obj,'action':action,'timestamp':timestamp,'exec':cmd, 'path':path})
    
    return info
# ref. [1]
def _get_infos_cadets(data: list) -> list:
    info = []
    for x in data:
        try:
            action = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['type']
        except:
            action = ''
        try:
            actor = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            actor = ''
        try:
            obj = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject']['com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            obj = ''
        try:
            timestamp = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['timestampNanos']
        except:
            timestamp = ''
        try:
            cmd = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['properties']['map']['exec']
        except:
            cmd = ''
        try:
            path = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObjectPath']['string']
        except:
            path = ''
        try:
            path2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2Path']['string']
        except:
            path2 = ''
        try:
            obj2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2']['com.bbn.tc.schema.avro.cdm18.UUID']
            info.append({'actorID':actor,'objectID':obj2,'action':action,'timestamp':timestamp,'exec':cmd, 'path':path2})
        except:
            pass
        
        info.append({'actorID':actor,'objectID':obj,'action':action,'timestamp':timestamp,'exec':cmd, 'path':path})
    
    return info
# ref. [1]
def _get_infos_trace(data: list) -> list:
    info = []
    for x in data:
        try:
            action = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['type']
        except:
            action = ''
        try:
            actor = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            actor = ''
        try:
            obj = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject']['com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            obj = ''
        try:
            timestamp = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['timestampNanos']
        except:
            timestamp = ''
        try:
            cmd = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['properties']['map']['exec']
        except:
            cmd = ''
        try:
            path = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObjectPath']['string']
        except:
            path = ''
        try:
            path2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2Path']['string']
        except:
            path2 = ''
        try:
            obj2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2']['com.bbn.tc.schema.avro.cdm18.UUID']
            info.append({'actorID':actor,'objectID':obj2,'action':action,'timestamp':timestamp,'exec':cmd, 'path':path2})
        except:
            pass
        
        info.append({'actorID':actor,'objectID':obj,'action':action,'timestamp':timestamp,'exec':cmd, 'path':path})
    
    return info
# ref. [1]
def _get_infos_fivedirections(data: list) -> list:
    info = []
    for x in data:
        try:
            action = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['type']
        except:
            action = ''
        try:
            actor = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            actor = ''
        try:
            obj = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject']['com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            obj = ''
        try:
            timestamp = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['timestampNanos']
        except:
            timestamp = ''
        try:
            cmd = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['properties']['map']['exec']
        except:
            cmd = ''
        try:
            path = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObjectPath']['string']
        except:
            path = ''
        try:
            path2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2Path']['string']
        except:
            path2 = ''
        try:
            obj2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2']['com.bbn.tc.schema.avro.cdm18.UUID']
            info.append({'actorID':actor,'objectID':obj2,'action':action,'timestamp':timestamp,'exec':cmd, 'path':path2})
        except:
            pass
        
        info.append({'actorID':actor,'objectID':obj,'action':action,'timestamp':timestamp,'exec':cmd, 'path':path})
    
    return info

# ref. [1], [2]
def _add_attributes(dataset_name: str, d,p):
    with open(p, 'r') as f:
        data = [json.loads(x) for x in f if "EVENT" in x]
    
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            info = _get_infos_theia(data)
        case SupportedDataset.CADETS3.value:
            info = _get_infos_cadets(data)
        case SupportedDataset.TRACE3.value:
            info = _get_infos_trace(data)
        case SupportedDataset.FIVEDIRECTIONS3.value:
            info = _get_infos_fivedirections(data)
        case _:
            raise_unsupported_dataset(dataset_name)
    
    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)
    return d.merge(rdf,how='inner',on=['actorID','objectID','action','timestamp']).drop_duplicates()


# ref. [1]
def prep_dataframe(dataset_name: str, file_processed_txt: str, file_for_adding_attributes_from_json: str):
    with open(file_processed_txt, "r") as f:
        data_raw = f.read().split('\n')
    data_raw = [line.split('\t') for line in data_raw]
    data_frame = pd.DataFrame (data_raw, columns = ['actorID','actor_type','objectID','object','action','timestamp'])
    data_frame = data_frame.dropna()
    data_frame.sort_values(by='timestamp', ascending=True, inplace=True)

    data_frame = _add_attributes(dataset_name, data_frame, file_for_adding_attributes_from_json)
    
    return data_frame


####################

