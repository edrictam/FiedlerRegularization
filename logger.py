# helper to write logs
def logToFile(filename, result_string):
    file = open(filename, "a+")
    file.write(result_string)
    file.close()
