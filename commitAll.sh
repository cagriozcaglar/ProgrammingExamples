# git config --system core.editor emacs
#git add --all .
git commit -m "$1"
randomText="origin`cat /dev/urandom | tr -cd 'a-f0-9' | head -c 6`"
git remote add ${randomText} https://github.com/cagriozcaglar/ProgrammingExamples.git
git remote -v
git push ${randomText} master
