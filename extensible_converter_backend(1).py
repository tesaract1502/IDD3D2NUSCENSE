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
        
        # Read sample_annotation.json to link annotations to instances
        sample_annot_path = os.path.join(dataloader.annot_out, 'sample_annotation.json')
        sample_annotations = []
        if os.path.exists(sample_annot_path):
            try:
                with open(sample_annot_path, 'r') as f:
                    sample_annotations = json.load(f)
            except Exception:
                pass
        
        # Read sample.json to get frame tokens
        sample_path = os.path.join(dataloader.annot_out, 'sample.json')
        samples = []
        if os.path.exists(sample_path):
            try:
                with open(sample_path, 'r') as f:
                    samples = json.load(f)
            except Exception:
                pass
        
        # IDD3D to nuScenes category mapping
        idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car',
            'Truck': 'vehicle.truck',
            'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle',
            'MotorcyleRider': 'vehicle.motorcycle',  # Handle typo in IDD3D
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
        
        # Read annotation data to get frame information
        annot_data = dataloader.read_annotations()
        if not annot_data:
            loghandler.log("No annotations found", "warning")
            return
        
        frame_ids = sorted(annot_data.keys())
        loghandler.log(f"Processing {len(frame_ids)} frames (01000-01099)", "info")
        
        # Track unique instances across all frames
        instance_tracker = {}  # obj_id -> {instance_token, category_token, annotations[], frames[]}
        
        # First pass: collect all unique objects and their annotations
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
                        # Generate instance token for this unique object
                        instance_token = uuid.uuid4().hex
                        
                        # Get category token
                        category_name = idd3d_to_nuscenes_categories.get(
                            obj_type, 
                            f'movable_object.{obj_type.lower()}'
                        )
                        category_token = category_map.get(category_name, uuid.uuid4().hex)
                        
                        instance_tracker[obj_id] = {
                            'instance_token': instance_token,
                            'category_token': category_token,
                            'annotation_tokens': [],
                            'frame_ids': []
                        }
                    
                    # Track frame appearance
                    instance_tracker[obj_id]['frame_ids'].append(frame_id)
            
            except Exception as e:
                loghandler.log(f"Error processing label {frame_id}: {str(e)}", "warning")
        
        # Second pass: link sample_annotation tokens to instances
        for annot in sample_annotations:
            instance_token = annot.get('instance_token')
            annot_token = annot.get('token')
            
            # Find matching instance by instance_token
            for obj_id, data in instance_tracker.items():
                if data['instance_token'] == instance_token:
                    data['annotation_tokens'].append(annot_token)
                    break
        
        # Create instance entries with all required fields
        instances = []
        for obj_id, data in instance_tracker.items():
            annotation_tokens = data['annotation_tokens']
            frame_ids = data['frame_ids']
            
            instance = {
                'token': data['instance_token'],
                'category_token': data['category_token'],
                'first_annotation_token': annotation_tokens[0] if annotation_tokens else '',
                'last_annotation_token': annotation_tokens[-1] if annotation_tokens else '',
                'first_frame_token': frame_ids[0] if frame_ids else '',
                'last_frame_token': frame_ids[-1] if frame_ids else ''
            }
            instances.append(instance)
        
        # Save instance.json
        out_path = os.path.join(dataloader.annot_out, 'instance.json')
        with open(out_path, 'w') as f:
            json.dump(instances, f, indent=2)
        
        loghandler.log(f"Instance file created with {len(instances)} instances", "success")
