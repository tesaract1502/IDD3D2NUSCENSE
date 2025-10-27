class IDD3DInstanceConverter(BaseConverter):
    """Generate nuScenes instance.json from IDD3D label files"""
    
    def __init__(self):
        super().__init__("instance")
    
    def run(self, dataloader: 'IDD3DDataLoader', loghandler: 'LogHandler'):
        import uuid
        
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
        
        # IDD3D to nuScenes category mapping
        idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car',
            'Truck': 'vehicle.truck',
            'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle',
            'MotorcyleRider': 'vehicle.motorcycle',
            'Bicycle': 'vehicle.bicycle',
            'Auto': 'vehicle.auto',
            'Person': 'human.pedestrian.adult',
            'Rider': 'human.pedestrian.rider',
            'Animal': 'animal',
            'TrafficLight': 'static_object.trafficlight',
            'TrafficSign': 'static_object.trafficsign',
            'Pole': 'static_object.pole',
            'OtherVehicle': 'vehicle.other',
            'Misc': 'movable_object.debris'
        }
        
        # Read annotation data
        annot_data = dataloader.read_annotations()
        if not annot_data:
            loghandler.log("No annotations found", "warning")
            return
        
        frame_ids = sorted(annot_data.keys())
        loghandler.log(f"Processing {len(frame_ids)} frames", "info")
        
        # Track unique instances: obj_id -> instance_data
        instance_tracker = {}
        
        # Process all frames (01000 to 01099)
        for frame_id in frame_ids:
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
                        # Generate instance token
                        instance_token = f"instance_{obj_id}"
                        
                        # Get category token
                        category_name = idd3d_to_nuscenes_categories.get(
                            obj_type, 
                            f'movable_object.{obj_type.lower()}'
                        )
                        category_token = category_map.get(category_name, f"cat_{obj_type}")
                        
                        instance_tracker[obj_id] = {
                            'instance_token': instance_token,
                            'category_token': category_token,
                            'first_frame': frame_id,
                            'last_frame': frame_id,
                            'first_annotation': f"obj_{obj_id}_{frame_id}",
                            'last_annotation': f"obj_{obj_id}_{frame_id}"
                        }
                    else:
                        # Update last frame and annotation
                        instance_tracker[obj_id]['last_frame'] = frame_id
                        instance_tracker[obj_id]['last_annotation'] = f"obj_{obj_id}_{frame_id}"
            
            except Exception as e:
                loghandler.log(f"Error processing label {frame_id}: {str(e)}", "warning")
        
        # Create instance entries matching the desired format
        instances = []
        for obj_id, data in sorted(instance_tracker.items()):
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
        
        loghandler.log(f"Instance file created with {len(instances)} instances", "success")
