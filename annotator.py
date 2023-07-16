import os
import tkinter
from nbv_utils import get_paths, read_json, write_json, read_pickle
from nbv_utils import get_node_id, parse_node_id
from PIL import Image, ImageTk
import cv2
import distinctipy
import numpy as np

def get_path_data(data_dir):
    path_ret = get_paths(data_dir, indices=None, use_filter_segs=True, use_filter_indices=True)
    image_inds, left_paths, _, _, _, seg_paths = path_ret
    
    return image_inds, left_paths, seg_paths

def get_clusters(data_dir):
    clusters_path = os.path.join(data_dir, 'associations', 'clusters.json')
    if not os.path.exists(clusters_path):
        raise RuntimeError('clusters path does not exist: ' + clusters_path)

    clusters = read_json(clusters_path)
    return clusters

def get_cluster_colors(clusters):
    fruitlet_ids = list(clusters["clusters"])
    num_fruitlets = len(fruitlet_ids)
    colors = distinctipy.get_colors(num_fruitlets)

    color_dict = {}
    for i in range(num_fruitlets):
        color = colors[i]
        color = ([int(255*color[0]), int(255*color[1]), int(255*color[2])])
        color_dict[int(fruitlet_ids[i])] = color

    return color_dict

def get_seg_image(im_path, seg_path, image_ind, clusters, cluster_colors):
    im = cv2.imread(im_path)
    segmentations = read_pickle(seg_path)

    seg_coords = {}

    for seg_id in range(len(segmentations)):
        node_id = get_node_id(image_ind, seg_id)
        if not node_id in clusters['fruitlets']:
            continue

        seg_inds, _ = segmentations[seg_id]
        cluster_num = int(clusters['fruitlets'][node_id])
        color = cluster_colors[cluster_num]

        im[seg_inds[:, 0], seg_inds[:, 1]] = color

        y, x = np.median(seg_inds, axis=0)
        y = int(np.round(y))
        x = int(np.round(x))
        seg_coords[seg_id] = [y, x]

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(im)

    return pil_image, seg_coords

def get_fruitlet_coords(fruitlet_id, clusters, seg_path, image_ind):
    node_ids = clusters["clusters"][str(fruitlet_id)]

    found = False
    for node_id in node_ids:
        node_image_ind, node_seg_ind = parse_node_id(node_id)
        if node_image_ind == image_ind:
            found = True
            break
    
    if not found:
        return False, None, None
    
    segmentations = read_pickle(seg_path)
    seg_inds, _ = segmentations[node_seg_ind]
        
    y, x = np.median(seg_inds, axis=0)
    y = int(np.round(y))
    x = int(np.round(x))

    return True, y, x

class Annotate():
    def __init__(self, data_dir, resize=False):
        self.data_dir = data_dir

        self.should_quit = None
        self.quit_type = None

        self.should_delete = None
        self.should_save = None
        self.ind_index = None
        self.prev_ind_index = None
        self.num_images = None

        self.curr_filename = None
        self.annotation_dict = None

        self.label_mode = None
        self.fruitlet_num = None

        self.resize=resize

    def annotate(self):
        # set up the gui
        window = tkinter.Tk()
        #window = tkinter.Toplevel()
        window.bind("<Key>", self.event_action)
        window.bind("<Button-1>", self.event_action_click)
        # get the list of jsons
        image_inds, im_paths, seg_paths = get_path_data(self.data_dir)
        sorted_inds = np.argsort(image_inds).tolist()
        image_inds = [image_inds[i] for i in sorted_inds]
        im_paths = [im_paths[i] for i in sorted_inds]
        seg_paths = [seg_paths[i] for i in sorted_inds]

        self.clusters = get_clusters(self.data_dir)
        cluster_colors = get_cluster_colors(self.clusters)
        annotation_path = os.path.join(self.data_dir, "gt_labels.json")

        self.should_quit = False
        self.quit_type = None

        self.ind_index = 0
        self.prev_ind_index = None
        self.num_images = len(image_inds)
        self.label_mode = False

        while not self.should_quit:
            self.should_save = False
            self.should_delete = False


            if self.ind_index == self.prev_ind_index:
                pass
            elif os.path.exists(annotation_path):
                self.annotation_dict = read_json(annotation_path)
            else:
                self.annotation_dict = {"forward": {},
                                        "backward": {}}

            self.prev_ind_index = self.ind_index

            image_ind = image_inds[self.ind_index]
            im_path = im_paths[self.ind_index]
            seg_path = seg_paths[self.ind_index]

            window_title = os.path.basename(self.data_dir) + '-' + str(image_ind)
            window.title(window_title)  
            picture, seg_coords = get_seg_image(im_path, seg_path, image_ind, 
                                                self.clusters, cluster_colors)
            self.latest_seg_coords = seg_coords
            self.latest_image_ind = image_ind
            if self.resize:
                picture = picture.resize((720, 540))

            tk_picture = ImageTk.PhotoImage(picture)
            picture_width = picture.size[0]
            picture_height = picture.size[1]
            window.geometry("{}x{}+100+100".format(picture_width, picture_height))
            image_widget = tkinter.Label(window, image=tk_picture)
            image_widget.place(x=0, y=0, width=picture_width, height=picture_height)

            for key in self.annotation_dict["forward"]:
                succ, y, x = get_fruitlet_coords(self.annotation_dict["forward"][key], self.clusters, 
                                                 seg_path, image_ind)

                if not succ:
                    continue

                if self.resize:
                    x = x / 2
                    y = y / 2

                num_text = tkinter.Label(window, text=str(key), font=("Helvetica", 8))
                relx = x / picture_width
                rely = y / picture_height
                num_text.place(anchor=tkinter.N, relx=relx, rely=rely)

            # wait for events
            if self.resize:
                window.geometry("720x540")
            window.mainloop()

            assert not (self.should_save and self.should_delete)

            if self.should_save:
                write_json(annotation_path, self.annotation_dict)

            if self.should_delete:
                if os.path.exists(annotation_path):
                    os.remove(annotation_path)

        window.destroy()
        return self.quit_type

    def event_action(self, event):
        character = event.char
        
        if character == 'q':
            self.should_quit = True
            self.quit_type = 'left'
            event.widget.quit()
        elif character == 'e':
            self.should_quit = True
            self.quit_type = 'right'
            event.widget.quit()
        elif character == 'w':
            self.should_quit = True
            self.quit_type = 'hard'
            event.widget.quit()
        elif character == 's':
            self.should_save = True
            self.should_delete = False
            self.prev_ind_index = None
            event.widget.quit()
        elif character == 'b':
            self.should_save = False
            self.should_delete = True
            self.prev_ind_index = None
            event.widget.quit()
        elif character in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            self.label_mode = True
            self.fruitlet_num = int(character)
        elif character == 'c':
            self.label_mode = False
            self.fruitlet_num = None
        elif character == 'k':
            self.label_mode = False
            self.fruitlet_num = None
            self.prev_ind_index = None
            event.widget.quit()
        elif character == 'a':
            if self.ind_index > 0:
                self.ind_index -= 1
                event.widget.quit()
        elif character == 'd':
            if self.ind_index < self.num_images - 1:
                self.ind_index += 1
                event.widget.quit()

    def event_action_click(self, event):
        if not self.label_mode:
            return

        x = event.x
        y = event.y

        if self.resize:
            x = 2*x
            y = 2*y

        min_dist = None
        for key in self.latest_seg_coords:
            y_c, x_c = self.latest_seg_coords[key]

            dist = np.linalg.norm([y_c - y, x_c - x])
            if (min_dist is None) or (dist < min_dist):
                min_dist = dist
                node_id = get_node_id(self.latest_image_ind, key)

        if min_dist is not None:
            fruitlet_id = int(self.clusters['fruitlets'][node_id])

            fruitlet_id_str = str(fruitlet_id)
            fruitlet_num_str = str(self.fruitlet_num)
            del_back = False
            del_for = False

            if fruitlet_id_str in self.annotation_dict["backward"]:
                prev_gt_val = self.annotation_dict["backward"][fruitlet_id_str]
                del_back = True

            if fruitlet_num_str in self.annotation_dict["forward"]:
                prev_cluster_val = self.annotation_dict["forward"][fruitlet_num_str]
                del_for = True

            if del_back:
                del self.annotation_dict["forward"][str(prev_gt_val)]
            if del_for:
                del self.annotation_dict["backward"][str(prev_cluster_val)]


            self.annotation_dict["forward"][str(self.fruitlet_num)] = fruitlet_id
            self.annotation_dict["backward"][fruitlet_id_str] = self.fruitlet_num


        self.label_mode = False
        self.fruitlet_num = None

        event.widget.quit()