import os
import json
import torch
import datasets
from pprint import pprint
from dataclasses import dataclass, field
from transformers import HfArgumentParser, is_torch_npu_available
from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder
from pyserini.output_writer import get_output_writer, OutputFormat


@dataclass
class ModelArgs:
    encoder: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'Name or path of encoder'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add instruction?'}
    )
    query_instruction_for_retrieval: str = field(
        default=None,
        metadata={'help': 'query instruction for retrieval'}
    )
    pooling_method: str = field(
        default='cls',
        metadata={'help': "Pooling method. Avaliable methods: 'cls', 'mean'"}
    )
    normalize_embeddings: bool = field(
        default=True,
        metadata={'help': "Normalize embeddings or not"}
    )
    queries: str = field(
        default="./data/test_query_LJP+simcrime_4ele_unique.json",
        metadata={'help': "queries"}
    )
    type: str = field(
        default="+typecrime",
        metadata={'help': "Normalize embeddings or not"}
    )


@dataclass
class EvalArgs:
    languages: str = field(
        default="en",
        metadata={'help': 'Languages to evaluate. Avaliable languages: ar de en es fr hi it ja ko pt ru th zh', 
                  "nargs": "+"}
    )
    index_save_dir: str = field(
        default='./corpus-index',
        metadata={'help': 'Dir to index and docid. Corpus index path is `index_save_dir/{encoder_name}/{lang}/index`. Corpus ids path is `index_save_dir/{encoder_name}/{lang}/docid` .'}
    )
    result_save_dir: str = field(
        default='./search_results',
        metadata={'help': 'Dir to saving search results. Search results will be saved to `result_save_dir/{encoder_name}/{lang}.txt`'}
    )
    threads: int = field(
        default=1,
        metadata={'help': 'Maximum threads to use during search'}
    )
    hits: int = field(
        default=1000,
        metadata={'help': 'Number of hits'}
    )
    overwrite: bool = field(
        default=False,
        metadata={'help': 'Whether to overwrite embedding'}
    )


def get_query_encoder(model_args: ModelArgs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif is_torch_npu_available():
        device = torch.device("npu")
    else:
        device = torch.device("cpu")
    model = AutoQueryEncoder(
        encoder_dir=model_args.encoder,
        device=device,
        pooling=model_args.pooling_method,
        l2_norm=model_args.normalize_embeddings
    )
    return model


def check_languages(languages):
    if isinstance(languages, str):
        languages = [languages]
    avaliable_languages = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']
    for lang in languages:
        if lang not in avaliable_languages:
            raise ValueError(f"Language `{lang}` is not supported. Avaliable languages: {avaliable_languages}")
    return languages

def deal_input(record, type, fact=None, item=None):
    if type == "common":
        assert fact != None
        return fact
    elif type == "4ele":
        return load_input(record)
    elif type == "+crime" or type=="+typecrime":
        return load_input(record, add_crime=True)


def load_input(data, add_crime=False):
    def dict_to_string(d, i):
        return f"\t罪名{str(i)}: " + ', '.join(f'{key}: {value}' for key, value in d.items())

    try:
        if "json" in data:
            data = data.split("json")[-1]
            data = data.split("```")[0]
        data = json.loads(data)
        crime = ""
        crime2 = "\t罪名:"
        i = 1
        if type(data) == list:
            for item in data:
                if '犯罪的四要件' in item:
                    crime += dict_to_string(item["犯罪的四要件"], i)
                    if add_crime == True:
                        crime2 += f"{item['罪名']}\t"
                    i += 1
                else:
                    for key in item:
                        crime += dict_to_string((item[key]["犯罪的四要件"]), i)
                        if add_crime == True:
                            crime2 += f"{item[key]['罪名']}\t"
                        i += 1
        else:
            for key in data:
                crime += dict_to_string(data[key]["犯罪的四要件"], i)
                if add_crime == True:
                    crime2 += f"{data[key]['罪名']}\t"
                i += 1
        if add_crime == True:
            return crime + crime2
        else:
            return crime
    except Exception:
        print(Exception)
        return ""


def get_queries_and_qids(lang: str,  type, queries, split: str='test',add_instruction: bool=False, query_instruction_for_retrieval: str=None):
    dataset = datasets.load_dataset('json',
                               data_files=queries)['train']
    queries = []
    qids = []
    for data in dataset:
        qids.append(str(data['id']))
        query = deal_input(data["query_4element"], type, data["fact"], data)
        #query=data["fact"][:512]
        queries.append(query)
    if add_instruction and query_instruction_for_retrieval is not None:
        queries = [f"{query_instruction_for_retrieval}{query}" for query in queries]

    return queries, qids


def save_result(search_results, result_save_path: str, qids: list, max_hits: int):
    output_writer = get_output_writer(result_save_path, OutputFormat(OutputFormat.TREC.value), 'w',
                                      max_hits=max_hits, tag='Faiss', topics=qids,
                                      use_max_passage=False,
                                      max_passage_delimiter='#',
                                      max_passage_hits=1000)
    with output_writer:
        for topic, hits in search_results:
            output_writer.write(topic, hits)


def main():
    parser = HfArgumentParser([ModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    eval_args: EvalArgs
    
    languages = check_languages(eval_args.languages)
    type=model_args.type

    if model_args.encoder[-1] == '/':
        model_args.encoder = model_args.encoder[:-1]
    
    query_encoder = get_query_encoder(model_args=model_args)
    
    encoder = model_args.encoder
    if os.path.basename(encoder).startswith('checkpoint-'):
        encoder = os.path.dirname(encoder) + '_' + os.path.basename(encoder)
    
    print("==================================================")
    print("Start generating search results with model:")
    print(model_args.encoder)

    print('Generate search results of following languages: ', languages)
    for lang in languages:
        print("**************************************************")
        print(f"Start searching results of {lang} ...")
        
        result_save_path = os.path.join(eval_args.result_save_dir, os.path.basename(encoder), f"{lang+type}.txt")
        if not os.path.exists(os.path.dirname(result_save_path)):
            os.makedirs(os.path.dirname(result_save_path))
        
        if os.path.exists(result_save_path) and not eval_args.overwrite:
            print(f'Search results of {lang} already exists. Skip...')
            continue
        
        index_save_dir = os.path.join(eval_args.index_save_dir, os.path.basename(encoder)+type, lang)
        if not os.path.exists(index_save_dir):
            raise FileNotFoundError(f"{index_save_dir} not found")
        searcher = FaissSearcher(
            index_dir=index_save_dir,
            query_encoder=query_encoder
        )
        
        queries, qids = get_queries_and_qids(
            lang=lang,
            type=type,
            queries=model_args.queries,
            split='test', 
            add_instruction=model_args.add_instruction, 
            query_instruction_for_retrieval=model_args.query_instruction_for_retrieval
        )
        
        search_results = searcher.batch_search(
            queries=queries,
            q_ids=qids,
            k=eval_args.hits,
            threads=eval_args.threads
        )
        search_results = [(_id, search_results[_id]) for _id in qids]
        
        save_result(
            search_results=search_results,
            result_save_path=result_save_path, 
            qids=qids, 
            max_hits=eval_args.hits
        )

    print("==================================================")
    print("Finish generating search results with model:")
    pprint(model_args.encoder)


if __name__ == "__main__":
    main()
