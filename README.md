### MODELLING TRAFFIC CONGESTION AND ESTIMATING LINK TRAVEL TIMES USING SPARSE PROBE DATA
------------------------------------------------------------------------------------

#### Description of Files in Repo
-----------------------------------

- **Functions.py** holds the bulk of the main functions used for dataset transformation, as well as any model related functions.

- Additional files, results and visualisations will be added when ready.

#### Traffic Estimation and Measuring Performance
--------------------------------------------------
One unfortunate aspect of using sparse probe data is the lack of access to a 'ground truth' congestion state. This makes
modelling challenging due to an additional difficulty in trying to measure how various models perform. As a result,
I choose to use cross-validation for this project, where I learn model parameters on the training set (80% of the data)
and then measure performance on the test-set (remaining 20% of the data). 

For the task of traffic estimation (which is a precursor to modelling), I have chosen to initally use a Kalman Filter.
Judging by the nature of our dataset (noisy observations of an underlying dynamically varying state), the use of this
filter seemed appropriate. 

I will also attempt to tackle the task of traffic estimation by means of representing the congestion in the city as a Gaussian
Process in terms of average speed. The continual-space varying nature of a Gaussian Process might provide deeper insight into
congestion related matters. Travel time distributions could then be found through integration of the speed distribution along the dimensions of a road.

#### Current Models Being Investigated
---------------------------------------
My initial modelling techniques treat the underlying dynamics of congestion as a linear dynamic stochastic process. Each
link is assumed to have a travel-time (time taken to traverse the full length of the road) which is distributed normally. I will treat the state of a link, *i*, at time *t*, to be conditionally independent of the states of non-neighbouring links at time *t-1* and the states of all links prior to time *t-1*. Finding the optimal dependency will be the subject of investigations yet to come. Machine learning techniques could be suitable for this.

Performance will be done through measuring the accuracy of the distribution generated for the travel time of each
link at each timestep of our model. Sampling techniques will be employed to see if measurements from the test set are
in accordance with our predicted distributions.



#### Background & Motivation for Work
--------------------------------------
Traffic congestion is a major problem cities around the world face. Congestion impacts
economic activity, decreases productivity and causes pollution: imposing many costs on
society. Congestion can be defined as the travel time on an arterial link (a city roadway),
which is in excess of that normally incurred under light, or free-flow travel conditions (Lomax
et al., 1997).

The latest technological advances and research have identified highly reliable methods of
monitoring the congestion states of arterial road networks. Google, for example, monitors
the speed of every phone whose GPS data it has access to and, in doing this, can build a
near perfect real-time estimation of congestion. Moreover, in more technologically advanced
cities, although more rare, computer vision techniques have been used alongside an ubiquitous
network of roadside cameras to do the same.

As strong as these current-day traffic monitoring systems are, the lack of an understanding
regarding the dynamics and variability of congestion leads to limited prediction capabilities
and thus is a subject of ongoing research throughout the world. Through a stronger under-
standing of the dynamics of congestion, cities can better plan and mitigate the inconveniences
surrounding prospective urban road operations, drivers can make more informed route choices
in advance, and traffic navigation companies like TomTom can improve their efforts in ad-
dressing the infamous route splitting problem (where drivers are given different routes to
take, for the same origin and destination, to overall reduce congestion in the network). Con-
sidering these potential benefits, trying to better understand the dynamics of traffic through
various modelling techniques will be a large facet of my work done this year. Moreover, if we
choose to represent congestion in terms of a travel-time probability distribution per link, we
will be in a better position to characterize the variability of congestion as well.

However, it is obvious that a precursor to the proposed work above is to have good,
reliable, real-time information regarding the congestion of links in the road network being
modelled. Whilst one may think that such a dataset is readily available, it is in fact the case
that outside Google's private GoogleMaps data, probe-vehicle data is the only significant data
source available pertaining to information about vehicles at the road-level. Such data mainly
refers to that coming from GPS-equipped fleet vehicles (such as taxis), that periodically
report their location and sometimes speed as they take their passengers from point A to B.
Considering the significant amount of this data available and the fact that cities around the
world are increasingly retrofitting their fleet vehicles with GPS systems, it is not surprising
that several research papers have attempted to accurately estimate the performance metric
of interest, congestion, using this auxiliary data source. Following this, the other
main aspect of the work I do this year will be adding to the body of work done on traffic inference algorithms,
targeted at transforming sparse probe-data into reliable congestion estimates.

#### The Dataset
-----------------
The dataset used in my project comes from a Kaggle challenge, listed by the European
Conference on Machine Learning and Principles and Practice of Knowledge Discovery in
Databases (https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data). The dataset describes a complete year (from 01/07/2013 to 30/06/2014) of
the trajectories of all 442 taxis running in the city of Porto, Portugal. This corresponds to
a total of 1710671 trips. The dataset contains several pieces of information regarding each
trip, such as taxi and caller id. The key piece of information we are interested in however, is
a field that contains a list of GPS coordinates that correspond to where the taxi was, every
15 seconds of a trip, from start to end. These GPS coordinates, alongside knowing the time
at which they were reported, should allow us to infer the congestion in the city at a given
time of day.

#### Transforming the Dataset
--------------------------
The dataset in its raw format is not of much use. What we are interested in knowing from
each consecutive pair of GPS points reported by a taxi is: which road was travelled on and
how much of that specific road was travelled. Furthermore, as GPS is not completely accurate
and generally comes with an error of up to 10m, we need to account for this error in our
models. The process I used in mapping the GPS reportings to their corresponding links is a
process known officially as `map-matching'.

Using Python, the drivable roads of Porto were downloaded and stored in the form of a
networkx multidigraph (a graph data-structure). This resulted in a network of over 13000
links (note that a link should ideally correspond to a length of road between two intersections)
which is too many to model. Furthermore, considering that our GPS reports are spaced 15
seconds apart, it does not make sense to model travel-time pdfs (probability density functions)
for roads that take much less time than 15 seconds to travel (choose 10 seconds as the limit
to be safe). Considering these things, we choose not to model the travel-time pdfs for all
roads in Porto, but only a particular subset which we will call the `links of interest'.
The speed limit on inner city roads in Porto is 50mph and we are only interested in roads
that take a minimum of 10 seconds to traverse (to limit irreducible modelling errors). If we
assume that under standard traffic conditions that the average speed of vehicles is 30mph,
this gives us a cut-off length of 130m as the minimum link length we are interested in (which
seems reasonable). Using this cut-off, we are left with 3368 links to model, which is much
more appropriate.

Another complication arises due the possibility of a taxi traversing several links in a 15
second interval (which was found to be quite common). In such cases, I needed to implement a
path-inference algorithm to identify the most likely path taken between the GPS points. Due
to 15 seconds being quite a fine temporal scale, we can assume that there is little uncertainty
in the path determined by the path inference algorithm.

Overall, the entire map-matching and path-inference algorithm converts the series of GPS
coordinates for each trip into a series of objects which hold data regarding which link was
traversed (does not need to be a link of interest), what proportion of the total length of that
link was traversed, and when the measurement was reported.

#### Important Papers Referenced
-------------------------------
- A. Hofleitner et al., "Learning the Dynamics of Arterial Traffic From Probe Data Using a
Dynamic Bayesian Network". 2012

- Xianyuan Zhan et al., "Urban link travel time estimation using large-scale taxi data with partial
information". 2013


