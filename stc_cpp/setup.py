from setuptools import find_packages, setup

package_name = 'stc_cpp'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='farhan',
    maintainer_email='theoriginalhybrid9@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'offline = stc_cpp.planner_offline:main',
            'online = stc_cpp.planner_online:main',
            'incremental = stc_cpp.incremental_planner_online:main',
            'tester_path = stc_cpp.tester_path:main',
            'test = stc_cpp.test:main',
        ],
    },
)
