import argparse
import huggingface_hub as hh
import os

MODELS_REPO = 'yilong2001/bert_cls_example'

def download_model(args):
    model_name = args.model_name
    size = args.size    
    filename = f'ggml-model-{size}.bin'
    hh.hf_hub_download(repo_id=MODELS_REPO, filename=f'{model_name}/{filename}', repo_type='model', revision='main', local_dir='.', local_dir_use_symlinks=False)
    print(f'{filename} downloaded successfully')

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description='Download GGML models')
    parser.set_defaults(func=lambda args: parser.print_help())
    subparsers = parser.add_subparsers()

    download_parser = subparsers.add_parser('download', help='Download a model')
    download_parser.add_argument('model_name', help='Name of the model')
    download_parser.add_argument('size', choices=['f32', 'f16'], help='Size of the model')
    download_parser.set_defaults(func=download_model)

    args = parser.parse_args()
    args.func(args)
