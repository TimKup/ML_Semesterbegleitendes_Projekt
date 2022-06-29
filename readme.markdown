# Konfigurationanleitung

### Folgende Pfade in der Datei config.py müssen angepasst werden, damit der Code im Skript main.py lauffähig ist: <br><br>

- <strong>OUTPUT_LABELING --> </strong>Pfad, welche den Speicherort für die gelabelten Daten angibt. Muss angepasst werden, damit die Funktionen label_data() aufgerufen werden kann.
Entspricht gleichzeitig dem Input-Pfad für die Funktionen prepeare_classification_data bzw.p repeare_regression_data (siehe INPUT_MODELS). <br><br>

### Optional:<br><br>

- <strong>OUTPUT_DIR --></strong> Gibt das Verzeichnis an, in dem die gelabelten Datensätze (Trainings- und Testdaten) abgespeichert werden sollen. Notwendig für die Funktion label_data().

- <strong>DATA_DIR --></strong> Muss auf die Originaldaten zeigen (Ausplittung in Learning_set und Test_set); Notwendig, falls Funktion create_data() ausgeführt werden soll.

- <strong>OUTPUT_TIME_DOMAIN --> </strong>Zeigt auf den Ordner, in den die extrahierten Merkmale pro Bearing für die Zeitsignalanalyse abgespeichert werden sollen. 
WICHTIG: In diesm Ordnern müssen jeweils die Ordner "Lernsets" und "Testsets" vorhanden sein.
Notwendig, falls Funktion create_data() ausgeführt werden soll.

- <strong>OUTPUT_FREQUENCY_DOMAIN --></strong> Zeigt auf den Ordner, in den die extrahierten Merkmale pro Bearing für die Frequenzanalyse abgespeichert werden sollen. 
WICHTIG: In diesm Ordnern müssen jeweils die Ordner "Lernsets" und "Testsets" vorhanden sein.
Notwendig, falls Funktion create_data() ausgeführt werden soll.