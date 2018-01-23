unzip *.zip
rm *.zip
mv *.png ../images/
cd ../..
git add .
git commit -m 'update'
git push -u origin master
