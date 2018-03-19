sudo su
DATA_DIR=/home/ubuntu/neuralEditor/neural-editor-data
REPO_DIR=/home/ubuntu/neuralEditor/neural-editor
export TEXTMORPH_DATA=$DATA_DIR
python run_docker.py --root --gpu 0

