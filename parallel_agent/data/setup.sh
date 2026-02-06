sudo apt update
sudo apt install -y aria2
bash hfd.sh BytedTsinghua-SIA/hotpotqa --dataset --tool aria2c -x 10 --local-dir ~/ruler_from_memagent

pip install datasets
# python gen_gsm_infinite.py 
# python gen_ruler_mini.py 
source ~/azureml_job_env.sh
cp -r $ZHIYUHE/qa_eval_more ~
