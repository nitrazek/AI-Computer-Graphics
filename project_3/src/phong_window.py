import os
import json
import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44

from base_window import BaseWindow


class PhongWindow(BaseWindow):

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frame = 0
        np.random.seed(42)  # Set a fixed seed for reproducibility

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model_matrix = self.program["model_matrix"]
        self.material_diffuse = self.program["material_diffuse"]
        self.material_shininess = self.program["material_shininess"]
        self.light_position = self.program["light_position"]
        self.camera_position = self.program["camera_position"]

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        if self.frame >= 3000:
            self.wnd.close()
            return

        config = (
            np.random.uniform(-5, 5), # model x
            np.random.uniform(-5, 5), # model y
            np.random.uniform(-5, 5), # model z
            np.random.uniform(0, 255),  # r diffuse
            np.random.uniform(0, 255),  # g diffuse
            np.random.uniform(0, 255),  # b diffuse
            np.random.uniform(3, 20),   # shininess
            np.random.uniform(-20, 20), # light x
            np.random.uniform(-20, 20), # light y
            np.random.uniform(-20, 20)  # light z
        )
            
        model_translation = [config[0], config[1], config[2]] 
        
        material_diffuse = [config[3]/255.0, config[4]/255.0, config[5]/255.0] 
        material_shininess = config[6] 
        light_position = [config[7], config[8], config[9]] 

        camera_position = [0.0, 0.0, 20.0]
        
        model_matrix = Matrix44.from_translation(model_translation)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            camera_position,
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

        model_view_projection = proj * lookat * model_matrix

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())
        self.model_matrix.write(model_matrix.astype('f4').tobytes())
        self.material_diffuse.write(np.array(material_diffuse, dtype='f4').tobytes())
        self.material_shininess.write(np.array([material_shininess], dtype='f4').tobytes())
        self.light_position.write(np.array(light_position, dtype='f4').tobytes())
        self.camera_position.write(np.array(camera_position, dtype='f4').tobytes())

        self.vao.render()
        
        if self.output_path:
            # Save Image
            img = (
                Image.frombuffer('RGBA', self.wnd.size, self.wnd.fbo.read(components=4))
                     .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            )
            img.save(os.path.join(self.output_path, f'image_{self.frame:04}.png'))
            
            # Save JSON Labels
            label_data = {
                "model_translation": model_translation,
                "material_ambient": [76.0/255.0, 76.0/255.0, 76.0/255.0],
                "material_diffuse": material_diffuse, 
                "material_specular": [255.0/255.0, 255.0/255.0, 255.0/255.0], 
                "material_shininess": material_shininess,
                "light_position": light_position,
                "light_ambient": [25.0/255.0, 25.0/255.0, 25.0/255.0], 
                "light_diffuse": [255.0/255.0, 255.0/255.0, 255.0/255.0], 
                "light_specular": [255.0/255.0, 255.0/255.0, 255.0/255.0],
                "camera_position": camera_position
            }
            
            with open(os.path.join(self.output_path, f'image_{self.frame:04}.json'), 'w') as f:
                json.dump(label_data, f, indent=4)

            self.frame += 1