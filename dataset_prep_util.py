####################

# references:
# [1] https://github.com/DART-Laboratory/Flash-IDS
# [2] Our contribution

####################

import re
import os
import json
import pandas as pd

####################

#### Supported datasets ####
from enum import Enum
class SupportedDataset(Enum):
    THEIA3 = "theia"

def raise_unsupported_dataset(dataset_name: str):
    raise ValueError(f"Dataset {dataset_name} is not supported currently.")

####################

# ref. [1], [2]
def download_dataset(dataset_name: str):
    import gdown
    
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            urls = ["https://drive.google.com/file/d/10cecNtR3VsHfV0N-gNEeoVeB89kCnse5/view?usp=drive_link",
                    "https://drive.google.com/file/d/1Kadc6CUTb4opVSDE4x6RFFnEy0P1cRp0/view?usp=drive_link"]
            for url in urls:
                gdown.download(url, quiet=False, use_cookies=False, fuzzy=True)
            
            print(f"Dataset {dataset_name} has been downloaded sucessfully.")
        
        case _:
            raise_unsupported_dataset(dataset_name)

# ref. [1], [2]
def extract_uuid(dataset_name: str, line: str):
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
            return pattern_uuid.findall(line)
        
        case _:
            raise_unsupported_dataset(dataset_name)

# ref. [1], [2]
def extract_subject_type(dataset_name: str, line: str):
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            pattern_type = re.compile(r'type\":\"(.*?)\"')
            return pattern_type.findall(line)
        
        case _:
            raise_unsupported_dataset(dataset_name)

# ref. [1]
def show(file_path):
    print(f"Processing {file_path}")

# ref. [1], [2]
def extract_edge_info(dataset_name: str, line: str):
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
            pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
            pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
            pattern_type = re.compile(r'type\":\"(.*?)\"')
            pattern_time = re.compile(r'timestampNanos\":(.*?),')

            edge_type = extract_subject_type(dataset_name, line)[0]
            timestamp = pattern_time.findall(line)[0]
            src_id = pattern_src.findall(line)

            if len(src_id) == 0:
                return None, None, None, None, None

            src_id = src_id[0]
            dst_id1 = pattern_dst1.findall(line)
            dst_id2 = pattern_dst2.findall(line)

            if len(dst_id1) > 0 and dst_id1[0] != 'null':
                dst_id1 = dst_id1[0]
            else:
                dst_id1 = None

            if len(dst_id2) > 0 and dst_id2[0] != 'null':
                dst_id2 = dst_id2[0]
            else:
                dst_id2 = None

            return src_id, edge_type, timestamp, dst_id1, dst_id2
        
        case _:
            raise_unsupported_dataset(dataset_name)

# ref. [1], [2]
def process_data(dataset_name: str, file_path: str):
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            id_nodetype_map = {}
            notice_num = 1000000
            for i in range(100):
                now_path = file_path + '.' + str(i)
                if i == 0:
                    now_path = file_path
                if not os.path.exists(now_path):
                    break

                with open(now_path, 'r') as f:
                    show(now_path)
                    cnt = 0
                    for line in f:
                        cnt += 1
                        if cnt % notice_num == 0:
                            print(cnt)

                        if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line:
                            continue

                        if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line:
                            continue

                        if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line:
                            continue

                        uuid = extract_uuid(dataset_name, line)[0]
                        subject_type = extract_subject_type(dataset_name, line)

                        if len(subject_type) < 1:
                            if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                                id_nodetype_map[uuid] = 'MemoryObject'
                                continue
                            if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                                id_nodetype_map[uuid] = 'NetFlowObject'
                                continue
                            if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                                id_nodetype_map[uuid] = 'UnnamedPipeObject'
                                continue

                        id_nodetype_map[uuid] = subject_type[0]

            return id_nodetype_map
        
        case _:
            raise_unsupported_dataset(dataset_name)

# ref. [1], [2]
def process_edges(dataset_name: str, file_path: str, id_nodetype_map):
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            notice_num = 1000000
            not_in_cnt = 0

            for i in range(100):
                now_path = file_path + '.' + str(i)
                if i == 0:
                    now_path = file_path
                if not os.path.exists(now_path):
                    break

                with open(now_path, 'r') as f, open(now_path+'.txt', 'w') as fw:
                    cnt = 0
                    for line in f:
                        cnt += 1
                        if cnt % notice_num == 0:
                            print(cnt)

                        if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                            src_id, edge_type, timestamp, dst_id1, dst_id2 = extract_edge_info(dataset_name, line)

                            if src_id is None or src_id not in id_nodetype_map:
                                not_in_cnt += 1
                                continue

                            src_type = id_nodetype_map[src_id]

                            if dst_id1 is not None and dst_id1 in id_nodetype_map:
                                dst_type1 = id_nodetype_map[dst_id1]
                                this_edge1 = f"{src_id}\t{src_type}\t{dst_id1}\t{dst_type1}\t{edge_type}\t{timestamp}\n"
                                fw.write(this_edge1)

                            if dst_id2 is not None and dst_id2 in id_nodetype_map:
                                dst_type2 = id_nodetype_map[dst_id2]
                                this_edge2 = f"{src_id}\t{src_type}\t{dst_id2}\t{dst_type2}\t{edge_type}\t{timestamp}\n"
                                fw.write(this_edge2)
        
        case _:
            raise_unsupported_dataset(dataset_name)

# ref. [1], [2]
def run_data_processing(dataset_name: str):
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            os.system('tar -zxvf ta1-theia-e3-official-1r.json.tar.gz')
            os.system('tar -zxvf ta1-theia-e3-official-6r.json.tar.gz')
            
            path_list = ['ta1-theia-e3-official-1r.json', 'ta1-theia-e3-official-6r.json']

            for path in path_list:
                id_nodetype_map = process_data(path)
                process_edges(path, id_nodetype_map)

            os.system('cp ta1-theia-e3-official-1r.json.txt theia_train.txt')
            os.system('cp ta1-theia-e3-official-6r.json.8.txt theia_test.txt')
        
        case _:
            raise_unsupported_dataset(dataset_name)

# ref. [1], [2]
def add_attributes(dataset_name: str, d,p):
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            f = open(p)
            data = [json.loads(x) for x in f if "EVENT" in x]

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

            rdf = pd.DataFrame.from_records(info).astype(str)
            d = d.astype(str)

            return d.merge(rdf,how='inner',on=['actorID','objectID','action','timestamp']).drop_duplicates()
        
        case _:
            raise_unsupported_dataset(dataset_name)


# ref. [1], [2]
def prep_dataframe(dataset_name: str, test_file_processed_txt: str, file_for_adding_attributes_from_json: str):
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            # test_file_processed_txt = "theia_test.txt"
            # file_for_adding_attributes_from_json = "ta1-theia-e3-official-6r.json.8"
            
            test_file = open(test_file_processed_txt)
            data_raw = test_file.read().split('\n')
            data_raw = [line.split('\t') for line in data_raw]
            data_frame = pd.DataFrame (data_raw, columns = ['actorID', 'actor_type','objectID','object','action','timestamp'])
            data_frame = data_frame.dropna()
            data_frame.sort_values(by='timestamp', ascending=True,inplace=True)

            data_frame = add_attributes(dataset_name, data_frame, file_for_adding_attributes_from_json)
            
            return data_frame


####################

