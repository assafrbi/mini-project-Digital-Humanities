# Topic Modeling of Israeli Movies Synopses - Digital Humanities Mini Project

## Main Objective

In this project, we explore the hidden themes behind IMDB movie synopses, originating from Israel.
We are doing so by applying the topic modeling method, which allows us to discover topics from corpus, gain insights on movie topic trends and dynamics within the movies created by the Israeli film industry, through the years.

## The Relation to Digital Humanities
Out Project serves as a method of distant reading, since we represent our parsed metadata results in a graphical form.
We also make use of already digitized dataset - IMDB, to produce new data sets that can be accessed by others.
As a result, digital humanities metadata in the field of Israeli film industry is enriched.

## The Work Process and Technology

Cinemagoer - an IMDB API, and Web Scaping technique assisted in gathering movies dataset, which encludes IMDB ID, Title, Release Year, Genre and Synopses.
Then, we performed text preprocessing (such as removal of new lines, quotes, lowercase all text, stopwords) after which we used Topic Modeling - Genesim LDA tool.
The model evaluated topics for each document (aka record) by its Synopsis.
Finally, the evaluated data allowed us to visualize the results as Intertopic Distance Map, WordCloud and also gain information regarding the average year of movies assigned to each topic and the genres assosicated with them.

Tools list:

  * Programming Language
    * Pyhton
  * Topic modeling tool
    * Gensim LDA python package 
  * DB formatt
    * CSV
  * Graphic tools
    * pyLDAvis
    * Bag Of Words   
  * Presentation
    * Google Colab Notebook
     

## Final Product

A Github project and Notebook that enables users to access our findings:
* Dataset
* Code
* Information needed to understand and use the project
* The results -
 * A graphic representation in form of "bag of words" of each topic (i.e. the larger the font size of the word, the more popular it is among the dataset)
 * The genres assinged to each topic that revealed by the model
 * The average year of movies in each topic that revealed by the model
 * Movies correlation - topic vice

## Resources

* [IMDB](https://www.imdb.com)
* [Cinemagoer (aka IMDBpy)](https://imdbpy.readthedocs.io/en/latest/)
* [Text preprocessing Techniques](https://github.com/Deffro/text-preprocessing-techniques/blob/master/techniques.py)
* [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/readme.html)
* [Bag Of Words](https://infogram.com/?utm_source=infogram&utm_medium=webview&utm_campaign=header_logo&utm_content=a5dd0a63-73bc-43d1-8343-61e11a09aeda)
* [Gensim LDA](https://radimrehurek.com/gensim/index.html#install)
* [Topic Modeling](https://en.wikipedia.org/wiki/Topic_model)
* [Wikipedia](https://www.wikipedia.org)
