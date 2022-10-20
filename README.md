<br>

**Films: An Experiment**

<br>

### Data

* [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/)

<br>

**``ratings.csv``**

RangeIndex: 25000095 entries, 0 to 25000094

&nbsp; | column | type
:--- | :--- | :---
0 |  userId | int64
1 |  movieId | int64
2 |  rating | float64
3 | timestamp | int64

<br>

**``movies.csv``**

&nbsp; | column | non-null | count | type
:--- | :--- |  :--- | :--- | :---
0 |  movieId | 62423 | non-null | int64
1 | title  |  62423 | non-null | object
2 | genres | 62423 | non-null | object

<br>

**``links.csv``**

&nbsp;  | column |  non-null | count | type
:--- | :--- |  :--- | :--- | :---
0 |  movieId | 62423 | non-null  | int64
1 |  imdbId  | 62423 | non-null  | int64
2 |  tmdbId  | 62316 | non-null  | float64

<br>
<br>

### Notes

Note, remember, that [pandas.DataFrame.max](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.max.html) skips 
NaN values by default.

<br>
<br>

### References

* [Multi-Armed Bandit for Recommendations, ``Kawale`` & ``Chow``, 2018](https://www.datacouncil.ai/talks/a-multi-armed-bandit-framework-for-recommendations-at-netflix)
* Confidence Interval
  * [Notes](https://www.itl.nist.gov/div898/handbook/prc/section1/prc14.htm)
  * [Normal Distribution: Critical Values](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3671.htm)
  * [scipy & critical values](www.statology.org/z-critical-value-python/)

<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>