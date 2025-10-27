class IDD3DCategoryConverter(BaseConverter):
    """Generate nuScenes category.json from IDD3D object types - FIXED MAPPING"""
    
    def __init__(self):
        super().__init__("category")
    
    def run(self, dataloader: 'IDD3DDataLoader', loghandler: 'LogHandler'):
        import uuid
        
        # FIXED: Proper IDD3D to nuScenes category mapping
        idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car',
            'Truck': 'vehicle.truck',
            'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle',
            'MotorcyleRider': 'vehicle.motorcycle',  # Handle typo - rider on motorcycle
            'Bicycle': 'vehicle.bicycle',
            'Auto': 'vehicle.auto',
            'Person': 'human.pedestrian.adult',
            'Pedestrian': 'human.pedestrian.adult',  # Alternative name
            'Rider': 'human.pedestrian.rider',
            'ScooterRider': 'human.pedestrian.rider',  # Scooter rider is a pedestrian type
            'Animal': 'animal',
            'TrafficLight': 'static_object.trafficlight',
            'TrafficSign': 'static_object.trafficsign',
            'Pole': 'static_object.pole',
            'OtherVehicle': 'vehicle.other',
            'Misc': 'movable_object.debris'
        }
        
        # Collect unique object types from all label files
        unique_obj_types = set()
        frame_ids = sorted(dataloader.read_annotations().keys())
        
        for frame_id in frame_ids:
            label_path = os.path.join(dataloader.label_dir, f'{frame_id}.json')
            if not os.path.exists(label_path):
                continue
            
            try:
                with open(label_path, 'r') as f:
                    label_objects = json.load(f)
                
                for obj in label_objects:
                    obj_type = obj.get('obj_type')
                    if obj_type:
                        unique_obj_types.add(obj_type)
            except Exception:
                pass
        
        loghandler.log(f"Found {len(unique_obj_types)} unique object types in dataset", "info")
        
        # Create category entries with PROPER mapping
        categories = []
        category_tokens = {}  # Track tokens to avoid duplicates
        
        for obj_type in sorted(unique_obj_types):
            # Map IDD3D type to nuScenes category name
            category_name = idd3d_to_nuscenes_categories.get(obj_type, f'movable_object.{obj_type.lower()}')
            
            # Avoid duplicate categories
            if category_name not in category_tokens:
                category_token = uuid.uuid4().hex
                category_tokens[category_name] = category_token
                
                category = {
                    'token': category_token,
                    'name': category_name,
                    'description': f'{obj_type} category from IDD3D'
                }
                categories.append(category)
                loghandler.log(f"Mapped {obj_type} -> {category_name}", "info")
        
        # Save category.json
        out_path = os.path.join(dataloader.annot_out, 'category.json')
        with open(out_path, 'w') as f:
            json.dump(categories, f, indent=2)
        
        loghandler.log(f"Category file created with {len(categories)} categories", "success")
