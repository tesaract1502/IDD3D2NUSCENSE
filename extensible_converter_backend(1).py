class IDD3DInstanceConverter(BaseConverter):
    """Generate nuScenes instance.json using sweeps folder images"""
    
    def __init__(self):
        super().__init__("instance")
    
    def run(self, dataloader: 'IDD3DDataLoader', loghandler: 'LogHandler'):
        import uuid
        import os
        
        # Read category.json to get category tokens
        category_path = os.path.join(dataloader.annot_out, 'category.json')
        category_map = {}  # category_name -> token
        if os.path.exists(category_path):
            try:
                with open(category_path, 'r') as f:
                    categories = json.load(f)
                    for cat in categories:
                        category_map[cat['name']] = cat['token']
            except Exception:
                pass
        
        # Define sweeps directory structure
        sweeps_dir = os.path.join(dataloader.out_data, 'sweeps')
        camera_channels = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]
        
        # Check if sweeps directory exists
        if not os.path.exists(sweeps_dir):
            loghandler.log(f"Sweeps directory not found: {sweeps_dir}", "error")
            return
        
        loghandler.log(f"Reading images from sweeps directory: {sweeps_dir}", "info")
        
        # Collect image files from each camera
        camera_images = {}
        for cam in camera_channels:
            cam_dir = os.path.join(sweeps_dir, cam)
            if os.path.exists(cam_dir):
                images = sorted([f for f in os.listdir(cam_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                camera_images[cam] = images
                loghandler.log(f"Found {len(images)} images in {cam}", "info")
            else:
                loghandler.log(f"Camera directory not found: {cam}", "warning")
                camera_images[cam] = []
        
        # Determine the maximum number of images across all cameras
        max_images = max(len(imgs) for imgs in camera_images.values()) if camera_images else 0
        
        if max_images == 0:
            loghandler.log("No images found in sweeps directory", "error")
            return
        
        loghandler.log(f"Processing {max_images} timesteps across {len(camera_channels)} cameras", "info")
        
        # Read label files to get object information
        annot_data = dataloader.read_annotations()
        if not annot_data:
            loghandler.log("No annotations found", "warning")
            return
        
        frame_ids = sorted(annot_data.keys())
        
        # Track unique instances: obj_id -> instance_data
        instance_tracker = {}
        instance_counter = 0
        
        # Process each timestep (matching image index across all cameras)
        for timestep_idx in range(max_images):
            # For each timestep, we process the corresponding frame
            # Match timestep to frame_id (e.g., timestep 0 -> frame 01000, timestep 1 -> frame 01001)
            if timestep_idx < len(frame_ids):
                frame_id = frame_ids[timestep_idx]
                
                # Read label file for this frame
                label_path = os.path.join(dataloader.label_dir, f'{frame_id}.json')
                if not os.path.exists(label_path):
                    continue
                
                try:
                    with open(label_path, 'r') as f:
                        label_objects = json.load(f)
                    
                    for obj in label_objects:
                        obj_id = obj.get('obj_id')
                        obj_type = obj.get('obj_type')
                        
                        if not obj_id or not obj_type:
                            continue
                        
                        if obj_id not in instance_tracker:
                            # Generate new instance token
                            instance_counter += 1
                            instance_token = f"instance_{instance_counter}"
                            
                            # Get category token
                            category_token = category_map.get(f'vehicle.{obj_type.lower()}', 
                                                             category_map.get(f'human.pedestrian.{obj_type.lower()}',
                                                             category_map.get(f'static_object.{obj_type.lower()}',
                                                             f"cat_{obj_type}")))
                            
                            instance_tracker[obj_id] = {
                                'instance_token': instance_token,
                                'category_token': category_token,
                                'first_timestep': timestep_idx,
                                'last_timestep': timestep_idx,
                                'first_frame': frame_id,
                                'last_frame': frame_id,
                                'first_annotation': f"obj_{obj_id}_{frame_id}",
                                'last_annotation': f"obj_{obj_id}_{frame_id}"
                            }
                            
                            loghandler.log(f"New instance {instance_token} detected for object {obj_id} ({obj_type})", "info")
                        else:
                            # Update last occurrence
                            instance_tracker[obj_id]['last_timestep'] = timestep_idx
                            instance_tracker[obj_id]['last_frame'] = frame_id
                            instance_tracker[obj_id]['last_annotation'] = f"obj_{obj_id}_{frame_id}"
                
                except Exception as e:
                    loghandler.log(f"Error processing label {frame_id}: {str(e)}", "warning")
        
        # Create instance entries
        instances = []
        for obj_id, data in sorted(instance_tracker.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
            instance = {
                "instance_token": data['instance_token'],
                "category_token": data['category_token'],
                "first_annotation_token": data['first_annotation'],
                "last_annotation_token": data['last_annotation'],
                "first_frame_token": f"frame_{data['first_frame']}",
                "last_frame_token": f"frame_{data['last_frame']}"
            }
            instances.append(instance)
        
        # Save instance.json
        out_path = os.path.join(dataloader.annot_out, 'instance.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(instances, f, indent=2)
        
        loghandler.log(f"Instance file created with {len(instances)} unique instances", "success")
        loghandler.log(f"Instances tracked across {max_images} timesteps from sweeps", "success")
