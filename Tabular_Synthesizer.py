"""Example using YData's regular data synthesizer - Local Version."""

from ydata.connectors import LocalConnector
from ydata.dataset.filetype import FileType
from ydata.metadata import Metadata
from ydata.synthesizers.regular.model import RegularSynthesizer

import os
os.environ['YDATA_LICENSE_KEY'] = '74ff0c2a-ae55-41ba-bb00-976bee030b68'

if __name__ == "__main__":

    # init the local connector
    connector = LocalConnector()

    # Read the file from local storage
    data = connector.read_file(
        path="./distract.csv",  # Local path to your CSV
        file_type=FileType.CSV
    )

    # Instantiate a synthesizer
    distract_synth = RegularSynthesizer()

    # calculating the metadata
    metadata = Metadata(dataset=data)

    # fit model to the provided data
    distract_synth.fit(X=data,
                     metadata=metadata,
                     condition_on='DRDISTRACT')  # Change to your target column

    # Generate data samples by the end of the synth process
    synth_sample = distract_synth.sample(n_samples=1000,
                                       balancing=True)

    # Write the sample to local storage
    connector.write_file(
        data=synth_sample,
        path="./synth_distract.csv",  # Local output path
        file_type=FileType.CSV,
    )

    # Store the synthesizer model
    #cardio_synth.save(path="./model.pkl")

    # Load and Sample (demonstrates reusing the saved model)
    #model = RegularSynthesizer.load(path="./model.pkl")
    #res = model.sample(100)
    #print(res.to_pandas().head())
