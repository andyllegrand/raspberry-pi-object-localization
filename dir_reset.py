import os

# be super careful with this!!
def reset_directory(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                reset_directory(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


reset_directory("/Users/andylegrand/PycharmProjects/objloc_ras_pi/output")