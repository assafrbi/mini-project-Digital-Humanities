# Topic Modeling of Israeli Movies Synopses - Digital Humanities Mini Project

## Main Objective

In this project, we will explore the hidden themes behind IMDB movie synopses, originating from Israel.
We are particularly interested in applying the topic modeling method to discover topics from corpus.
This will allow us to gain some insight into movie topic trends and interesting dynamics within the movies created by the Israeli film industry.

## The Relation to Digital Humanities

We intend the project to serve as a method of distant reading, since we hope to represent our parsed metadata results in a graphical form.
We will also make use of already digitized datasets, such as IMDB and Wikidata, to produce new data sets that can be accessed by others.
As a result, digital humanities metadata in the field of Israeli film industry will be enriched.

## Work Plan & Technology

IMDB and Wikidata/Wikipedia APIs will assist us in gathering movies synopses datasets.
Then, we will perform text preprocessing (such as removal of new lines, quotes, lowercase all text etc.) after which we will use topic modeling LDA tool.
The selected tool will evaluate topics for each document (synopses).
Finally, the evaluated data will allow us visualize the results.

A list of optional tools:

  * Programming Language
    * Pyhton
  * Topic modeling tools
    * Google code
    * Gensim LDA python package
    * Sklearn python package
    * Mallet  
  * DB storage
    * CSV
    * XML
    * JSON
    * YAML
    
  * Graphic tools
    * pyLDAvis
    * Bag Of Words   
  * Presentation
    * Wix site
     


## Final Product

A website will enable users to access our findings and contain all the information they need to understand and use the project, such as:
  * About tab - the idea of the project and the process
  * Download dataset tab
  * The results -
    * A graphic representation in form of "bag of words" of the topics statistics, i.e. the larger the font size of the word (topic), the more popular it is among the data set we create.
    * Movies correlation - topic vice
    * Statistics regarding the difference between the genre and the topic revealed by the model

## Resources

* [IMDB](https://www.imdb.com)
* [IMDBpy](https://github.com/dormbase/IMDBpy)
* [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page)
* [Wikipedia](https://www.wikipedia.org)
* [Text preprocessing Techniques](https://github.com/Deffro/text-preprocessing-techniques/blob/master/techniques.py)
* [Google code](https://code.google.com/archive/p/topic-modeling-tool/)
* [Mallet](https://mimno.github.io/Mallet/topics.html)
* [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/readme.html)
* [Bag Of Words](https://infogram.com/?utm_source=infogram&utm_medium=webview&utm_campaign=header_logo&utm_content=a5dd0a63-73bc-43d1-8343-61e11a09aeda)
* [Sklearn LDA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
* [Gensim LDA](https://radimrehurek.com/gensim/index.html#install)
* [Topic Modeling](https://en.wikipedia.org/wiki/Topic_model)
