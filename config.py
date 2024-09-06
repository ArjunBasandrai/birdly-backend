class Config:
    max_size = 1024 * 1024 * 10; # 10MB
    image_size = (480, 480)
    model_path = 'model.h5'
    class_names_path = 'cnames.txt'
    
    high_prob = 0.75
    medium_prob = 0.5
    low_prob = 0.25

    high_k = 5
    medium_k = 3
    low_k = 1

