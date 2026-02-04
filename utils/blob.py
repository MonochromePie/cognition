import numpy as np

def blobize(image, edge_image):
    
    visited = np.zeros(image.shape[:2], dtype=np.bool)
    
    def is_skippable(i, j):
        if i < 0 or i >= image.shape[0] or j < 0 or j >= image.shape[1]:
            return True
        if edge_image[i][j] >= 128:
            # Is an edge pixel, skip
            visited[i][j] = True
            return True
        return visited[i][j]
    
    jobs = []
    blob_images = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if is_skippable(i, j):
                continue
            jobs.append((i, j))
            blob_image = np.zeros(image.shape, dtype=np.uint8)
            # Using while instead of recursion
            while len(jobs) > 0:
                i, j = jobs.pop()
                if is_skippable(i, j):
                    continue
                blob_image[i][j] = image[i][j]
                visited[i][j] = True
                jobs.append((i + 1, j))
                jobs.append((i - 1, j))
                jobs.append((i, j + 1))
                jobs.append((i, j - 1))
            blob_images.append(blob_image)
    return blob_images