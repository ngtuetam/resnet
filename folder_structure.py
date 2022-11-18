import os
"""
Input:
    - n (int) : the number of class
Output:
    Folder Structure:
    main_directory/
        ---data/
            ---train/
                ---class_a/
                    ---a_image_1.jpg/
                    ---a_image_2.jpg/
                ---class_b/
                    ---b_image_1.jpg/
                    ---b_image_2.jpg/
            ---validation/
                ---class_a/
                    ---a_image_1.jpg/
                    ---a_image_2.jpg/
                ---class_b/
                    ---b_image_1.jpg/
                    ---b_image_2.jpg/
"""
try:
    number = int(input('Number of classes: '))
    if number > 1:
        name = []
        for type in range(number):
            name.append(input("Name of class {}:".format(type+1)))
        os.mkdir('data')
        os.chdir('data')
        path = ['train', 'validation']
        for i in path:
            os.mkdir(i)
            os.chdir('{}/'.format(i))
            for j in name:
                os.mkdir(j)
            os.chdir('../')
        print('Data including {} classes folder have been created'.format(number))
    else:
        print('Class must be greater than 1')
except ValueError:
    print('Enter a number!')
except FileExistsError:
    print('File existed')
