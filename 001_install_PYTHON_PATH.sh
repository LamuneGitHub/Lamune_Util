echo "아래 명령을 실행 하시요"
echo "source ./001_install_PYTHON_PATH.sh"

val_cur_dir=$( pwd )
export PYTHONPATH=$val_cur_dir

echo  PYTHONPATH=$val_cur_dir


echo -e export PYTHONPATH=$val_cur_dir >> ~/.zshrc
