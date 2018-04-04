# Identifying Superheroes from product images
This was a hackathon hosted on CrowdAnalytix

Name of the hackathon: [Identifying Superheroes from product images](https://www.crowdanalytix.com/contests/identifying-superheroes-from-product-images)



## Overview
While using machine learning to perform image recognition is currently one of the most popular use cases, in some cases, the existing large-scale models are too broad to be effective for specific business use cases. In this contest we will use a data driven approach to identify the “superheroes” in an image (fashion product images)

## Approach

I used the approach as defined by Jeremy Howard in his fantastic [fast.ai course*](http://course.fast.ai/)

_*I was on Lesson 2 of the course when I participated in the hackathon_

The steps that are defined to build a world class model are:
1) Enable data augementation and precompute =True
2) Use lr_find() to identify the highest learning rate when the loss is still improving
3) Train last layer from precomputed activation for 1-2 epochs
4) Train last layer with data augmentation but precompute = False
5) Unfreeze all layers
6) Set earlier layers learning rate to 3x-9x lower than the higer layer
7) Use lr_find() again
8) Train full network with cycle_mult = 2 untill overfitting

I was only able to train the model until steps 3-4, whenever I tried to train the other layers, I kept running into a memory problem. Probably because of the lower configuration or some other reason I couldn't fathom! 

Other steps that I tried with some success:

** Clean the training database. There were some obvious mix-ups, which I corrected

** Used a web-scraping (via the Bing API) to get more training samples. This approach didn't work as my system was simply hanging up...


## Leaderboard

Public Leaderboard: 0.81355 (36th out of 375 solvers)

Private Leaderboard: 0.75831 (34th of 375 solvers)

In my crossvalidation I was getting a score of 0.74, so the score on the Private Leaderboard was not surprising.
It also shows that fast.ai has a very good cross validation mechanism!


