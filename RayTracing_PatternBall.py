import taichi as ti
import numpy as np
from ray_tracing_models import Ray, Camera, Hittable_list, Sphere, PI, random_in_unit_sphere, refract, reflect, reflectance, random_unit_vector
ti.init(arch=ti.gpu)

# Canvas
aspect_ratio = 1.0
image_width = 800
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

# Rendering parameters
samples_per_pixel = 4
max_depth = 10
sample_on_unit_sphere_surface = True

PI = 3.14159265

@ti.kernel
def render():
    for i, j in canvas:
        u = (i + ti.random()) / image_width
        v = (j + ti.random()) / image_height
        color = ti.Vector([0.0, 0.0, 0.0])
        for n in range(samples_per_pixel):
            ray = camera.get_ray(u, v)
            color += ray_color(ray)
        color /= samples_per_pixel
        canvas[i, j] += color

# Path tracing
@ti.func
def ray_color(ray):
    color_buffer = ti.Vector([0.0, 0.0, 0.0])
    brightness = ti.Vector([1.0, 1.0, 1.0])
    scattered_origin = ray.origin
    scattered_direction = ray.direction
    p_RR = 0.8
    for n in range(max_depth):
        if ti.random() > p_RR:
            break
        is_hit, hit_point, hit_point_normal, front_face, material, color = scene.hit(Ray(scattered_origin, scattered_direction))
        if is_hit:
            if material == 0:
                color_buffer = color * brightness
                break
            else:
                # Diffuse
                if material == 1:
                    target = hit_point + hit_point_normal
                    if sample_on_unit_sphere_surface:
                        target += random_unit_vector()
                    else:
                        target += random_in_unit_sphere()
                    scattered_direction = target - hit_point
                    scattered_origin = hit_point
                    brightness *= color
                # Metal and Fuzz Metal
                elif material == 2 or material == 4:
                    fuzz = 0.0
                    if material == 4:
                        fuzz = 0.4
                    scattered_direction = reflect(scattered_direction.normalized(),
                                                  hit_point_normal)
                    if sample_on_unit_sphere_surface:
                        scattered_direction += fuzz * random_unit_vector()
                    else:
                        scattered_direction += fuzz * random_in_unit_sphere()
                    scattered_origin = hit_point
                    if scattered_direction.dot(hit_point_normal) < 0:
                        break
                    else:
                        brightness *= color
                # Dielectric
                elif material == 3:
                    refraction_ratio = 1.5
                    if front_face:
                        refraction_ratio = 1 / refraction_ratio
                    cos_theta = min(-scattered_direction.normalized().dot(hit_point_normal), 1.0)
                    sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                    # total internal reflection
                    if refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random():
                        scattered_direction = reflect(scattered_direction.normalized(), hit_point_normal)
                    else:
                        scattered_direction = refract(scattered_direction.normalized(), hit_point_normal, refraction_ratio)
                    scattered_origin = hit_point
                    brightness *= color
                # Pattern
                elif material == 5:
                    v1 = (9*camera.cam_horizontal[None] - camera.cam_vertical[None])/10
                    v2 = (camera.cam_horizontal[None] + 9*camera.cam_vertical[None])/10
                    cos_theta = min(-scattered_direction.normalized().dot(hit_point_normal), 1.0)
                    if (
                        int(hit_point_normal.dot(v1)*10)%2 == 1
                        and int(hit_point_normal.dot(v2)*10)%2 == 1
                    ):
                        brightness *= color*5*cos_theta*cos_theta
                    else:
                        brightness *= (1-color)
                    scattered_direction = (reflect(scattered_direction.normalized(), hit_point_normal) 
                                            + 0.1 * random_unit_vector())
                    scattered_origin = hit_point
                elif material == 6:
                    v2 = (camera.cam_horizontal[None] - camera.cam_vertical[None])/2
                    v1 = (camera.cam_horizontal[None] + camera.cam_vertical[None])/2
                    th = ti.random()*0.8
                    if (
                        ti.sin(hit_point_normal.dot(v1)*50) > 0+th
                        and ti.sin(hit_point_normal.dot(v2)*50)> -0.5+th
                    ):
                        brightness *= color*0.9
                    else:
                        brightness *= (1-color)*0.8
                    scattered_direction = (reflect(scattered_direction.normalized(), hit_point_normal) 
                                            + 0.4 * random_unit_vector())
                    scattered_origin = hit_point
                elif material == 7:
                    v1 = camera.cam_vertical[None]
                    v2 = camera.cam_horizontal[None]
                    v3 = (camera.cam_horizontal[None] + camera.cam_vertical[None])/2
                    th = ti.random()*0.2
                    if (
                         th+0.3 > hit_point_normal.dot(v1) + ti.sin(hit_point_normal.dot(v2)*PI*3)*0.2 >th-0.1
                         or
                         th+0.3 > hit_point_normal.dot(v2) + ti.sin(hit_point_normal.dot(v1)*PI*3)*0.2 >th-0.1
                    ):
                        brightness *= color
                    else:
                        brightness *= (1-color)
                    scattered_direction = (reflect(scattered_direction.normalized(), hit_point_normal) 
                                            + 0.4 * random_unit_vector())
                    scattered_origin = hit_point
                
                brightness /= p_RR
    return color_buffer


if __name__ == "__main__":
    max_depth = 10
    samples_per_pixel = 4 #4
    sample_on_unit_sphere_surface = True
    scene = Hittable_list()

    # Light source
    # scene.add(Sphere(center=ti.Vector([0, 5.4, -1]), radius=3.0, material=0, color=ti.Vector([10.0, 10.0, 10.0])))
    scene.add(Sphere(center=ti.Vector([0, 5.4, -1]), radius=3.0, material=0, color=ti.Vector([10.0, 10.0, 10.0])))
    # Ground
    scene.add(Sphere(center=ti.Vector([0, -100.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # ceiling
    scene.add(Sphere(center=ti.Vector([0, 102.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # back wall
    scene.add(Sphere(center=ti.Vector([0, 1, 101]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # right wall
    scene.add(Sphere(center=ti.Vector([-101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.6, 0.0, 0.0])))
    # left wall
    scene.add(Sphere(center=ti.Vector([101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.0, 0.6, 0.0])))

    # Diffuse ball
    scene.add(Sphere(center=ti.Vector([0, -0.2, -1.5]), radius=0.3, material=1, color=ti.Vector([0.8, 0.3, 0.3])))
    # Metal ball
    scene.add(Sphere(center=ti.Vector([-0.8, 0.2, -1]), radius=0.7, material=5, color=ti.Vector([0.0, 0.4, 0.8])))
    # Glass ball
    scene.add(Sphere(center=ti.Vector([0.7, 0, -0.5]), radius=0.5, material=6, color=ti.Vector([0.0, 0.4, 0.8])))
    # Metal ball-2
    scene.add(Sphere(center=ti.Vector([0.6, -0.3, -2.0]), radius=0.4, material=7, color=ti.Vector([0.0, 0.4, 0.8])))
    # Pattern ball
    # scene.add(Sphere(center=ti.Vector([0.4, 1.5, -1.2]), radius=0.6, material=5, color=ti.Vector([0.0, 0.4, 0.8])))
    # scene.add(Sphere(center=ti.Vector([-0.4, 1.6, -1.3]), radius=0.5, material=6, color=ti.Vector([0.0, 0.4, 0.8])))



    camera = Camera()
    gui = ti.GUI("Ray Tracing", res=(image_width, image_height))
    canvas.fill(0)
    cnt = 0
    while gui.running:
        render()
        cnt += 1
        gui.set_image(np.sqrt(canvas.to_numpy() / cnt))
        gui.show()
        
        