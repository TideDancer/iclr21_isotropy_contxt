source venv/bin/activate

# run gen_embeds, run GPT on PTB dataset, collect results for GPT layer 0
mkdir embeds
mkdir embeds/ptb
python gen_embeds.py gpt ptb 0 --no_cuda --save_file gpt.layer.0.dict 

# run analysis
mkdir images
python analysis.py embeds/ptb/gpt.layer.0.dict --draw token --draw_token "['the','first','man']"

deactivate
