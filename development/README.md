<br>

Development Notes

<br>

### Development Environment

The virtual environment, locally named ``miscellaneous``, is outlined within 
the [energy repository](https://github.com/briefings/energy#development-notes).  To generate the 
dotfile that [``pylint``](https://pylint.pycqa.org/en/latest/user_guide/checkers/features.html)  - the
static code analyser - will use for analysis, run

````shell
pylint --generate-rcfile > .pylintrc
````

The requirements summary, using [filter.txt](../docs/filter.txt), is created via

````shell
pip freeze -r docs/filter.txt > requirements.txt
````

<br>
<br>

### Tools

**PyTest**

```shell
python -m pytest ...
```

<br>

**PyTest & Coverage**

```shell
python -m pytest --cov-report term-missing --cov src/data tests/data
```

<br>

**Pylint**

```shell
python -m pylint --rcfile .pylintrc src/data
```

Note that

```
logger.info('\n %s', data.info())
```

is preferred to

```
logger.info('\n{}'.format(data.info()))
```
<br>

**flake8**

```shell
# logic
python -m flake8 --count --select=E9,F63,F7,F82 --show-source 
          --statistics src/data
# complexity          
python -m flake8 --count --exit-zero --max-complexity=10 --max-line-length=127 
          --statistics src/data
```

<br>
<br>

### References

* Requests
  * https://docs.python-requests.org/en/master/index.html
* Pylint
  * http://pylint.pycqa.org/en/latest/user_guide/run.html#command-line-options
  * https://pylint.readthedocs.io/en/latest/technical_reference/features.html
  * [API Reference](https://docs.pytest.org/en/7.1.x/reference/reference.html)
  * [flags](https://docs.pytest.org/en/7.1.x/reference/reference.html#command-line-flags)
* [pytest](https://docs.pytest.org/en/7.1.x/contents.html)
* pytest & coverage
  * [about](https://pytest-cov.readthedocs.io/en/latest/)
  * [pytest --cov-report term-missing --cov src/directory tests/directory](https://pytest-cov.readthedocs.io/en/latest/reporting.html)
* Formatting
  * https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting

<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>