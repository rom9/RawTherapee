/*
 *  This file is part of RawTherapee.
 *
 *  Copyright (c) 2019 Alberto Romei <aldrop8@gmail.com>
 *
 *  RawTherapee is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  RawTherapee is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with RawTherapee.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <iomanip>

#include "filmnegative.h"

#include "editwidgets.h"
#include "eventmapper.h"
#include "options.h"
#include "rtimage.h"

#include "../rtengine/procparams.h"

namespace
{

Adjuster* createExponentAdjuster(AdjusterListener* listener, const Glib::ustring& label, double minV, double maxV, double defaultVal)
{
    Adjuster* const adj = Gtk::manage(new Adjuster(label, minV, maxV, 0.001, defaultVal));
    adj->setAdjusterListener(listener);
    adj->setLogScale(6, 1, true);

    if (adj->delay < options.adjusterMaxDelay) {
        adj->delay = options.adjusterMaxDelay;
    }

    adj->show();
    return adj;
}

}

FilmNegative::FilmNegative() :
    FoldableToolPanel(this, "filmnegative", M("TP_FILMNEGATIVE_LABEL"), false, true),
    EditSubscriber(ET_OBJECTS),
    evFilmNegativeExponents(ProcEventMapper::getInstance()->newEvent(FIRST, "HISTORY_MSG_FILMNEGATIVE_VALUES")),
    evFilmNegativeEnabled(ProcEventMapper::getInstance()->newEvent(FIRST, "HISTORY_MSG_FILMNEGATIVE_ENABLED")),
    evFilmNegativeMedians(ProcEventMapper::getInstance()->newEvent(FIRST, "HISTORY_MSG_FILMNEGATIVE_MEDIANS")),
    fnp(nullptr),
    greenExp(createExponentAdjuster(this, M("TP_FILMNEGATIVE_GREEN"), 0.3, 4, 1.5)),  // master exponent (green channel)
    redRatio(createExponentAdjuster(this, M("TP_FILMNEGATIVE_RED"), 0.3, 3, (2.04 / 1.5))), // ratio of red exponent to master exponent
    blueRatio(createExponentAdjuster(this, M("TP_FILMNEGATIVE_BLUE"), 0.3, 3, (1.29 / 1.5))), // ratio of blue exponent to master exponent
    spotgrid(Gtk::manage(new Gtk::Grid())),
    spotbutton(Gtk::manage(new Gtk::ToggleButton(M("TP_FILMNEGATIVE_PICK")))),
    recalcCropMedians(Gtk::manage(new Gtk::Button(M("TP_FILMNEGATIVE_CROP_MEDIANS"))))
{
    spotgrid->get_style_context()->add_class("grid-spacing");
    setExpandAlignProperties(spotgrid, true, false, Gtk::ALIGN_FILL, Gtk::ALIGN_CENTER);

    setExpandAlignProperties(spotbutton, true, false, Gtk::ALIGN_FILL, Gtk::ALIGN_CENTER);
    spotbutton->get_style_context()->add_class("independent");
    spotbutton->set_tooltip_text(M("TP_FILMNEGATIVE_GUESS_TOOLTIP"));
    spotbutton->set_image (*Gtk::manage (new RTImage ("color-picker-small.png")));

    recalcCropMedians->set_tooltip_text(M("TP_FILMNEGATIVE_CROP_MEDIANS_TOOLTIP"));

    // TODO make spot size configurable ?

    // Gtk::Label* slab = Gtk::manage (new Gtk::Label (M("TP_WBALANCE_SIZE")));
    // setExpandAlignProperties(slab, false, false, Gtk::ALIGN_START, Gtk::ALIGN_CENTER);

    // Gtk::Grid* wbsizehelper = Gtk::manage(new Gtk::Grid());
    // wbsizehelper->set_name("WB-Size-Helper");
    // setExpandAlignProperties(wbsizehelper, false, false, Gtk::ALIGN_START, Gtk::ALIGN_CENTER);

    // spotsize = Gtk::manage (new MyComboBoxText ());
    // setExpandAlignProperties(spotsize, true, false, Gtk::ALIGN_FILL, Gtk::ALIGN_CENTER);
    // spotsize->append ("2");
    // spotsize->set_active(0);
    // spotsize->append ("4");

    spotgrid->attach (*spotbutton, 0, 1, 1, 1);
    spotgrid->attach (*recalcCropMedians, 0, 2, 1, 1);
    // spotgrid->attach (*slab, 1, 0, 1, 1);
    // spotgrid->attach (*wbsizehelper, 2, 0, 1, 1);

    pack_start(*greenExp, Gtk::PACK_SHRINK, 0);
    pack_start(*redRatio, Gtk::PACK_SHRINK, 0);
    pack_start(*blueRatio, Gtk::PACK_SHRINK, 0);
    pack_start(*spotgrid, Gtk::PACK_SHRINK, 0);

    spotbutton->signal_toggled().connect(sigc::mem_fun(*this, &FilmNegative::editToggled));
    // spotsize->signal_changed().connect( sigc::mem_fun(*this, &WhiteBalance::spotSizeChanged) );

    recalcCropMedians->signal_clicked().connect(sigc::mem_fun(*this, &FilmNegative::recalcCropMediansClicked));

    // Editing geometry; create the spot rectangle
    Rectangle* const spotRect = new Rectangle();
    spotRect->filled = false;
    
    visibleGeometry.push_back(spotRect);

    // Stick a dummy rectangle over the whole image in mouseOverGeometry.
    // This is to make sure the getCursor() call is fired everywhere.
    Rectangle* const imgRect = new Rectangle();
    imgRect->filled = true;

    mouseOverGeometry.push_back(imgRect);
}

FilmNegative::~FilmNegative()
{
    for (auto geometry : visibleGeometry) {
        delete geometry;
    }

    for (auto geometry : mouseOverGeometry) {
        delete geometry;
    }
}

void FilmNegative::read(const rtengine::procparams::ProcParams* pp, const ParamsEdited* pedited)
{
    disableListener();

    if (pedited) {
        redRatio->setEditedState(pedited->filmNegative.redRatio ? Edited : UnEdited);
        greenExp->setEditedState(pedited->filmNegative.greenExp ? Edited : UnEdited);
        blueRatio->setEditedState(pedited->filmNegative.blueRatio ? Edited : UnEdited);
        set_inconsistent(multiImage && !pedited->filmNegative.enabled);
    }

    setEnabled(pp->filmNegative.enabled);
    redRatio->setValue(pp->filmNegative.redRatio);
    greenExp->setValue(pp->filmNegative.greenExp);
    blueRatio->setValue(pp->filmNegative.blueRatio);

    channelMedians[0] = pp->filmNegative.redMedian;
    channelMedians[1] = pp->filmNegative.greenMedian;
    channelMedians[2] = pp->filmNegative.blueMedian;

    enableListener();
}

void FilmNegative::write(rtengine::procparams::ProcParams* pp, ParamsEdited* pedited)
{
    pp->filmNegative.redRatio = redRatio->getValue();
    pp->filmNegative.greenExp = greenExp->getValue();
    pp->filmNegative.blueRatio = blueRatio->getValue();
    pp->filmNegative.redMedian = channelMedians[0];
    pp->filmNegative.greenMedian = channelMedians[1];
    pp->filmNegative.blueMedian = channelMedians[2];
    pp->filmNegative.enabled = getEnabled();

    if (pedited) {
        pedited->filmNegative.redRatio = redRatio->getEditedState();
        pedited->filmNegative.greenExp = greenExp->getEditedState();
        pedited->filmNegative.blueRatio = blueRatio->getEditedState();
        pedited->filmNegative.medians = channelMedians[0] != pp->filmNegative.redMedian
            || channelMedians[1] != pp->filmNegative.greenMedian
            || channelMedians[2] != pp->filmNegative.blueMedian;
        pedited->filmNegative.enabled = !get_inconsistent();
    }
}

void FilmNegative::setDefaults(const rtengine::procparams::ProcParams* defParams, const ParamsEdited* pedited)
{
    redRatio->setValue(defParams->filmNegative.redRatio);
    greenExp->setValue(defParams->filmNegative.greenExp);
    blueRatio->setValue(defParams->filmNegative.blueRatio);

    if (pedited) {
        redRatio->setDefaultEditedState(pedited->filmNegative.redRatio ? Edited : UnEdited);
        greenExp->setDefaultEditedState(pedited->filmNegative.greenExp ? Edited : UnEdited);
        blueRatio->setDefaultEditedState(pedited->filmNegative.blueRatio ? Edited : UnEdited);
    } else {
        redRatio->setDefaultEditedState(Irrelevant);
        greenExp->setDefaultEditedState(Irrelevant);
        blueRatio->setDefaultEditedState(Irrelevant);
    }
}

void FilmNegative::setBatchMode(bool batchMode)
{
    if (batchMode) {
        spotConn.disconnect();
        removeIfThere(this, spotgrid, false);
        ToolPanel::setBatchMode(batchMode);
        redRatio->showEditedCB();
        greenExp->showEditedCB();
        blueRatio->showEditedCB();
    }
}

void FilmNegative::adjusterChanged(Adjuster* a, double newval)
{
    if (listener) {
        if (a == redRatio || a == greenExp || a == blueRatio) {
            if (getEnabled()) {
                listener->panelChanged(
                    evFilmNegativeExponents,
                    Glib::ustring::compose(
                        "Ref=%1\nR=%2\nB=%3",
                        greenExp->getValue(),
                        redRatio->getValue(),
                        blueRatio->getValue()
                    )
                );
            }
        }
    }
}

void FilmNegative::enabledChanged()
{
    if (listener) {
        if (get_inconsistent()) {
            listener->panelChanged(evFilmNegativeEnabled, M("GENERAL_UNCHANGED"));
        }
        else if (getEnabled()) {
            listener->panelChanged(evFilmNegativeEnabled, M("GENERAL_ENABLED"));
        }
        else {
            listener->panelChanged(evFilmNegativeEnabled, M("GENERAL_DISABLED"));
        }
    }
}

void FilmNegative::setFilmNegProvider(FilmNegProvider* provider)
{
    fnp = provider;
}

void FilmNegative::setEditProvider(EditDataProvider* provider)
{
    EditSubscriber::setEditProvider(provider);
}

CursorShape FilmNegative::getCursor(int objectID) const
{
   return CSSpotWB;
}

bool FilmNegative::mouseOver(int modifierKey)
{
    EditDataProvider* const provider = getEditProvider();
    Rectangle* const spotRect = static_cast<Rectangle*>(visibleGeometry.at(0));
    spotRect->setXYWH(provider->posImage.x - 16, provider->posImage.y - 16, 32, 32);

    return true;
}

bool FilmNegative::button1Pressed(int modifierKey)
{
    EditDataProvider* const provider = getEditProvider();

    EditSubscriber::action = EditSubscriber::Action::NONE;

    if (listener) {
        refSpotCoords.push_back(provider->posImage);

        if (refSpotCoords.size() == 2) {
            // User has selected 2 reference gray spots. Calculating new exponents
            // from channel values and updating parameters.

            std::array<float, 3> newExps;
            if (fnp->getFilmNegativeExponents(refSpotCoords[0], refSpotCoords[1], newExps)) {
                disableListener();
                // Leaving green exponent unchanged, setting red and blue exponents based on
                // the ratios between newly calculated exponents.
                redRatio->setValue(newExps[0] / newExps[1]);
                blueRatio->setValue(newExps[2] / newExps[1]);
                enableListener();

                if (listener && getEnabled()) {
                    listener->panelChanged(
                        evFilmNegativeExponents,
                        Glib::ustring::compose(
                            "Ref=%1\nR=%2\nB=%3",
                            greenExp->getValue(),
                            redRatio->getValue(),
                            blueRatio->getValue()
                        )
                    );
                }
            }

            switchOffEditMode();
        }
    }

    return true;
}

bool FilmNegative::button1Released()
{
    EditSubscriber::action = EditSubscriber::Action::NONE;
    return true;
}

void FilmNegative::switchOffEditMode()
{
    refSpotCoords.clear();
    unsubscribe();
    spotbutton->set_active(false);
}

void FilmNegative::editToggled()
{
    if (spotbutton->get_active()) {
        subscribe();

        int w, h;
        getEditProvider()->getImageSize(w, h);

        // Stick a dummy rectangle over the whole image in mouseOverGeometry.
        // This is to make sure the getCursor() call is fired everywhere.
        Rectangle* const imgRect = static_cast<Rectangle*>(mouseOverGeometry.at(0));
        imgRect->setXYWH(0, 0, w, h);
    } else {
        refSpotCoords.clear();
        unsubscribe();
    }
}

void FilmNegative::recalcCropMediansClicked()
{
    if (listener) {

        disableListener();

        std::array<float, 3> newMedians = {};
        if (fnp->getFilmNegativeMedians(newMedians)) {

            channelMedians = newMedians;

            enableListener();

            if (listener && getEnabled()) {
                listener->panelChanged(
                    evFilmNegativeMedians,
                    Glib::ustring::compose("R=%1\nG=%2\nB=%3",
                        channelMedians[0], channelMedians[1], channelMedians[2]
                    )
                );
            }
        }

    }


}
