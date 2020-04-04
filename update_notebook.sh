#!/bin/bash

jupyter nbconvert --execute --to notebook --inplace cov.ipynb

git add cov.ipynb
git commit -m 'update notebook'
git push
