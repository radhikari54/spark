assembly: clean
	python setup.py build --build-lib build/lib
	unzip lib/py4j*.zip -d build/lib
	cd build/lib && zip -r ../pyspark-assembly.zip .

clean:
	- rm -r build
